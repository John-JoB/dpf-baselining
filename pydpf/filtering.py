from typing import Callable, Union, Dict
from .custom_types import Resampler, ImportanceKernel, ImportanceSampler
import torch
from torch import Tensor
from .utils import normalise
from .base import Module
from pydpf.resampling import SystematicResampler, SoftResampler, OptimalTransportResampler, StopGradientResampler, KernelResampler, VariationalResampler, SVGD_kernel_resampling, AuxiliaryResampler, MultinomialResampler
from .distributions import KernelMixture
from .model_based_api import FilteringModel
from .base import DivergenceError
from warnings import warn
from .conditional_resampling import ConditionalResampler
from copy import copy


"""
Python module for the core filtering algorithms. 

In pydpf a filtering algorithm is defined by it's sampling procedures rather than the underlying model. 
For example, in order sequential importance sampling, instead of providing a prior, dynamic kernel and observation model; 
one provides two procedures: importance sampling from the posterior at time zero and importance sampling from the posterior at subsequent time
steps. We motivate this choice in two respects, foremostly the flexibility provided with this design we permit the user to easily use 
whatever proposal strategy their use case calls for without the package getting in the way. Secondarily, it is more conceptually inline
with the set-up of most current DPF problems. One learns an algorithm that performs well on the desired metrics, not a model that is
close to the truth.

The downside to this strategy being it requires some extra work from the user in certain cases. For, example should the user want to try 
slightly differing filtering algorithms on the same underlying parameterisation, then it is fully on them to design their code to make that 
process easy.

Note on Callables: most Modules in this file take at least one Callable argument, these must be instantiated Module (or less preferably 
torch.nn.Module) classes with the forward() method defined if the function contains learnable parameters.

Note on data: data should be treated similarly abstractly to the algorithm/model. The data passed to the proposal routines at each timestep
should be whatever data is required to sample the desired posterior. In a vanilla (bootstrap) filtering scenario this the observation at 
that time-step, but it doesn't have to be in general. pydpf makes no distinction between observations (past, present or future) and 
exogenous variables, all are treated as non-random inputs.
"""

class SIS(Module):
    """
    SMC filters can, in general, be described as special cases of sequential importance sampling (SIS).
    We provide this generic SIS class that can be extended for a given use case, or used by directly supplying the relevant functions.
    SIS iteratively importance samples a Markov-Chain.
    An SIS algorithm is defined by supplying an initial distribution and a Markov kernel.
    """
    def __init__(self, *, initial_proposal: Callable = None, proposal: Callable =None):
        """
        Module that represents a sequential importance sampling (SIS) algorithm. A SIS algorthm is fully specified by its importance sampling
        procedures, the user should supply a proposal kernel that may depend on the time-step; and a special case for time 0.

        Notes
        -----
        This implementation is more general than the standard SIS algorithm. There is no independence requirements for the samples within a
        batch. This means that the particles can be drawn from an arbitrary joint distribution on depended on the data and the particles at
        the previous time-step. Both the usual particle filter and interacting multiple model particle filter are special
        cases of this algorithm. It's also possible to make the filters within a batch depend on each other, but don't do that.


        Parameters
        ----------
        initial_proposal: ImportanceKernelLikelihood
            A callable object that takes the number of particles and the data/observations at time-step zero and returns an importance sample
            of the posterior, i.e. particle position and log weights. Also returns the observation likelihood (if applicable).
        proposal: ImportanceKernelLikelihood
            A callable object that implements the proposal kernel. Takes the state and log weights at the previous time step,
            the discreet time index i.e. how many iterations the filter has run for; and the data/observations at the current time-step.
            And returns an importance sample of the posterior at the current time step, i.e. particle position and log weights.
            Also returns the observation likelihood (if applicable).
        """
        super().__init__()
        if initial_proposal is not None:
            self._register_functions(initial_proposal, proposal)


    def _register_functions(self, initial_proposal: Callable, proposal: Callable):
        self.initial_proposal = initial_proposal
        self.proposal = proposal
        self.aggregation_function = None

    @staticmethod
    def _get_time_data(t: int, **data: dict, ) -> dict:
        time_dict = {k:v[t] for k, v in data.items() if k != 'series_metadata' and k != 'state' and v is not None}
        time_dict['t'] = t
        if data['time'] is not None and t>0:
            time_dict['prev_time'] = data['time'][t-1]
        if data['series_metadata'] is not None:
            time_dict['series_metadata'] = data['series_metadata']
        return time_dict

    def forward(self, n_particles: int, time_extent: int, aggregation_function: Union[Dict[str, Module], Module], observation, *, gradient_regulariser: torch.autograd.Function = None, ground_truth = None, control = None, time = None, series_metadata = None) -> Tensor:
        """
        Run a forward pass of the SIS filter. To save memory during inference runs we allow the user to pass a function that takes a population
        of particles and processes this into an output for each time-step. For example, if the goal was the filtering mean then it would be
        wasteful to store the full population of the particles for every time-step. Memory is not saved during training, because the
        computation graph is stored.

        Parameters
        ----------.
        n_particles: int
            The number of particles to draw per filter.
        time_extent: int
            The maximum time-step to run to, including time 0, the filter will draw {time_extent + 1} importance sample populations.
        aggregation_function: Union[Dict[str, Module], Module]
            A module that's forward function processes the filtering outputs (the particle locations, the normalised log weights,
            the log sum of the unormalised weights, the data, the time-step) into an output per time-step.
            Or a string indexed dictionary of such items.
        data: Tensor
            The data needed to run the filter

        Returns
        -------
        output: Tensor or Dict[str, Tensor]
            The output of the filter, formed from stacking the output of aggregation_function for every time-step.
            Or if aggregation_function is a dictionary, a dictionary of these output Tensors one for each aggregation function.
        """
        #Register module should the aggregation function have learnable parameters
        log_N = torch.log(torch.tensor(n_particles, dtype=torch.float32, device=observation.device))
        if isinstance(aggregation_function,dict):
            self.aggregation_function = torch.nn.ModuleDict(aggregation_function)
            output_dict = True
        else:
            self.aggregation_function = aggregation_function
            output_dict = False

        if self.training or not torch.is_grad_enabled():
            gradient_regulariser = None
        gt_exists = False
        if ground_truth is not None:
            gt_exists = True
        time_data = self._get_time_data(0, observation = observation, control = control, time = time, series_metadata = series_metadata)
        state, weight, likelihood = self.initial_proposal(n_particles = n_particles, **time_data)
        likelihood = likelihood - log_N
        if output_dict:
            output = {}
            for name, function in aggregation_function.items():
                if gt_exists:
                    temp = function(state=state, weight=weight, likelihood=likelihood, ground_truth=ground_truth[0], **time_data)
                else:
                    temp = function(state=state, weight=weight, likelihood=likelihood, **time_data)
                output[name] = torch.empty((time_extent+1, *temp.size()), device = observation.device, dtype=torch.float32)
                output[name][0] = temp
        else:
            if gt_exists:
                temp = aggregation_function(state = state, weight = weight, likelihood = likelihood, ground_truth = ground_truth[0], **time_data)
            else:
                temp = aggregation_function(state = state, weight = weight, likelihood = likelihood, **time_data)
            output = torch.empty((time_extent+1, *temp.size()), device = observation.device, dtype=torch.float32)
            output[0] = temp
        for t in range(1, time_extent+1):
            try:
                time_data = self._get_time_data(t, observation = observation, control = control, time = time, series_metadata = series_metadata)
                prev_state = state
                prev_weight = weight
                state, weight, likelihood = self.proposal(prev_state = state, prev_weight = weight, **time_data)
                likelihood = likelihood - log_N
                if not gradient_regulariser is None:
                    state, weight = gradient_regulariser(state = state, weight = weight, prev_state= prev_state, prev_weight = prev_weight)
                if output_dict:
                    for name, function in aggregation_function.items():
                        if gt_exists:
                            output[name][t] = function(state=state, weight=weight, likelihood=likelihood, ground_truth=ground_truth[t], **time_data)
                        else:
                            output[name][t] = function(state=state, weight=weight,likelihood=likelihood, **time_data)
                else:
                    if gt_exists:
                        output[t] = aggregation_function(state=state, weight=weight, likelihood=likelihood, ground_truth = ground_truth[t], **time_data)
                    else:
                        output[t] = aggregation_function(state=state, weight=weight, likelihood=likelihood, **time_data)
            except DivergenceError as e:
                warn(f'Detected divergence at time-step {t} with message:\n    {e} \nStopping iteration early.')
                return output[:t-1]
        return output


class ParticleFilter(SIS):
    """
        Helper class for a common case of the SIS, the particle filter (Doucet and Johansen 2008), (Chopin and Papaspiliopoulos 2020).
        Applies a resampling step prior to sampling from the proposal kernel.
    """
    def __init__(self, resampler: Resampler = None, SSM: FilteringModel = None, use_REINFORCE:bool = False) -> None:
        """
        The standard particle filter is a special case of the SIS algorithm. We construct the particle filtering proposal by first
        resampling particles from their population, then applying a proposal kernel restricted such that the particles depend only on the
        population at the previous time-step through the particle at the same index.

        Parameters
        ----------
        resampler: Resampler
            The resampling algorithm to use. Takes teh state and log weights at the previous time-step and returns the state and log weights
            after resampling. Resampling algorithms must also return a third tensor. Used to report extra information about how the particles
            were chosen, in most cases the resampled indices; this is often useful for diagnostics. But, this basic implementation discards
            this tensor.
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        use_REINFORCE_for_proposal: bool
            Whether to use the REINFORCE estimator for the gradient due to the particle proposal process. Applying REINFORCE to only some components of the
            state space is not permitted with this API, such a use case would require a custom SIS process.
        """
        self.REINFORCE = use_REINFORCE

        super().__init__()
        if resampler is not None:
            self._register_functions(resampler=resampler, SSM=SSM)

    def _register_functions(self, resampler: Resampler, SSM: FilteringModel):
        self.SSM = SSM
        self.resampler = resampler

        if self.REINFORCE:
            if not hasattr(SSM.dynamic_model, 'log_density'):
                raise AttributeError("The dynamic model must implement a 'log_density' method for REINFORCE.")
            if not hasattr(SSM.prior_model, 'log_density'):
                raise AttributeError("The prior model must implement a 'log_density' method for REINFORCE.")

        if self.SSM.initial_proposal_model is None:
            if self.REINFORCE:
                def prior(n_particles, observation, **data):
                    state = self.SSM.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data).detach()
                    density = self.SSM.prior_model.log_density(state = state, **data)
                    weight = self.SSM.observation_model.score(state = state, observation = observation, **data) + density - density.detach()
                    return state, weight
            else:
                def prior(n_particles, observation, **data):
                    state = self.SSM.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                    weight = self.SSM.observation_model.score(state = state, observation = observation, **data)
                    return state, weight
        else:
            if self.REINFORCE:
                def prior(n_particles, observation, **data):
                    state = self.initial_proposal_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data).detach()
                    weight = (self.SSM.observation_model.score(state = state, observation = observation, **data)
                              - self.SSM.initial_proposal_model.log_density(state = state, observation = observation, **data).detach()
                              + self.SSM.prior_model.log_density(state = state, **data))
                    return state, weight
            else:
                def prior(n_particles, observation, **data):
                    state = self.initial_proposal_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                    weight = (self.SSM.observation_model.score(state = state, observation = observation, **data)
                              - self.SSM.initial_proposal_model.log_density(state = state, observation = observation, **data)
                              + self.SSM.prior_model.log_density(state = state, **data))
                    return state, weight

        if self.SSM.proposal_model is None:
            if self.REINFORCE:
                def prop(prev_state, prev_weight, observation, **data):
                    new_state = self.SSM.dynamic_model.sample(prev_state = prev_state, **data).detach()
                    density = self.SSM.dynamic_model.log_density(state=new_state, prev_state=prev_state, **data)
                    new_weight = prev_weight + self.SSM.observation_model.score(new_state, observation = observation, **data) + density - density.detach()
                    return new_state, new_weight
            else:
                def prop(prev_state, prev_weight, observation, **data):
                    new_state = self.SSM.dynamic_model.sample(prev_state = prev_state, **data)
                    new_weight = prev_weight + self.SSM.observation_model.score(new_state, observation = observation, **data)
                    return new_state, new_weight
        else:
            if self.REINFORCE:
                def prop(prev_state, prev_weight, observation, **data):
                    new_state = self.SSM.proposal_model.sample(prev_state = prev_state, **data).detach()
                    new_weight = (prev_weight + self.SSM.observation_model.score(state = new_state, observation = observation, **data)
                                  - self.SSM.proposal_model.log_density(state = new_state, prev_state = prev_state, observation = observation, **data).detach()
                                  + self.SSM.dynamic_model.log_density(state = new_state, prev_state = prev_state, **data))
                    return new_state, new_weight
            else:
                def prop(prev_state, prev_weight, observation, **data):
                    new_state = self.SSM.proposal_model.sample(prev_state = prev_state, **data)
                    new_weight = (prev_weight + self.SSM.observation_model.score(state = new_state, observation = observation, **data)
                                  - self.SSM.proposal_model.log_density(state = new_state, prev_state = prev_state, observation = observation, **data)
                                  + self.SSM.dynamic_model.log_density(state = new_state, prev_state = prev_state, **data))
                    return new_state, new_weight


        def initial_sampler(n_particles: int, **data):
            state, weight = prior(n_particles=n_particles, **data)
            weight, likelihood = normalise(weight)
            return state, weight, likelihood

        if isinstance(self.resampler, ConditionalResampler):
            def pf_sampler(prev_state, prev_weight, **data):
                resampled_x, resampled_w = self.resampler(prev_state, prev_weight, **data)
                state, weight = prior(prev_state=resampled_x, prev_weight=resampled_w, **data)
                try:
                    weight, likelihood = normalise(weight)
                except ValueError:
                    raise DivergenceError('Found batch where all weights are small.')
                return state, weight, torch.where(self.resampler.cache['mask'], likelihood, likelihood - normalise(resampled_w)[1])
        else:
            def pf_sampler(prev_state, prev_weight, **data):
                resampled_x, resampled_w = self.resampler(prev_state, prev_weight, **data)
                state, weight = prop(prev_state=resampled_x, prev_weight=resampled_w, **data)
                try:
                    weight, likelihood = normalise(weight)
                except ValueError:
                    raise DivergenceError('Found batch where all weights are small.')
                return state, weight, likelihood
        super()._register_functions(initial_sampler, pf_sampler)


class MarginalParticleFilter(SIS):
    def __init__(self, resampler: Resampler = None, SSM: FilteringModel = None, REINFORCE_method:str = 'none'):
        super().__init__()
        self.REINFORCE = REINFORCE_method


        super().__init__()
        if resampler is not None:
            self._register_functions(resampler=resampler, SSM=SSM)
        if resampler is not None:
            self._register_functions(resampler=resampler, SSM=SSM)

    def _register_functions(self, resampler: Resampler, SSM: FilteringModel):
        self.resampler = resampler
        self.SSM = SSM

        if self.REINFORCE == 'full':
            if not hasattr(SSM.prior_model, 'log_density'):
                raise AttributeError("The prior model must implement a 'log_density' method for full REINFORCE.")

        if not hasattr(SSM.dynamic_model, 'log_density'):
            raise AttributeError("The dynamic model must implement a 'log_density' method for the marginal particle filter.")

        identity = lambda x: x
        detach_fun = lambda x: x.detach()

        detach_full = identity
        detach_partial = identity

        if self.REINFORCE not in ['full', 'partial', 'none']:
            raise ValueError("Parameter REINFORCE_method must be 'full', 'partial' or 'none'.")

        if self.REINFORCE == "full":
            detach_full = detach_fun
        if self.REINFORCE == "partial":
            detach_partial = detach_fun


        if self.SSM.initial_proposal_model is None:
            if self.REINFORCE == 'full':
                def prior(n_particles, observation, **data):
                    state = self.SSM.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data).detach()
                    density = self.SSM.prior_model.log_density(state = state, **data)
                    weight = self.SSM.observation_model.score(state = state, observation = observation, **data) + density - density.detach()
                    return state, weight
            else:
                def prior(n_particles, observation, **data):
                    state = self.SSM.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                    weight = self.SSM.observation_model.score(state = state, observation = observation, **data)
                    return state, weight
        else:
            def prior(n_particles, observation, **data):
                state = detach_full(self.initial_proposal_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data))
                weight = (self.SSM.observation_model.score(state = state, observation = observation, **data)
                          - detach_full(self.SSM.initial_proposal_model.log_density(state = state, observation = observation, **data))
                          + self.SSM.prior_model.log_density(state = state, **data))
                return state, weight

        def initial_sampler(n_particles: int, **data):
            state, weight = prior(n_particles=n_particles, **data)
            weight, likelihood = normalise(weight)
            return state, weight, likelihood





        if self.SSM.proposal_model is None:
            def prop(prev_state, prev_weight, observation, **data):
                resampled_state, resampled_weight = self.resampler(prev_state, prev_weight, **data)
                state = self.SSM.dynamic_model.sample(prev_state=resampled_state, **data)
                used_weight = self.resampler.cache['used_weight']
                expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                dynamic_log_density = self.SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                weight = (torch.logsumexp(prev_weight.unsqueeze(1) + detach_partial(dynamic_log_density), dim=-1)
                          - detach_full(detach_partial(torch.logsumexp(used_weight.unsqueeze(1) + dynamic_log_density, dim=-1)))
                          + self.SSM.observation_model.score(state=state, observation=observation, **data))
                return state, weight, resampled_weight, dynamic_log_density, dynamic_log_density
        else:
            def prop(prev_state, prev_weight, observation, **data):
                resampled_state, resampled_weight = self.resampler(prev_state, prev_weight, **data)
                state = detach_full(self.SSM.proposal_model.sample(prev_state=resampled_state, **data))
                used_weight = self.resampler.cache['used_weight']
                expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                dynamic_log_density = self.SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                proposal_log_density = self.SSM.proposal_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                weight = (torch.logsumexp(prev_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                          - detach_full(torch.logsumexp(used_weight.unsqueeze(1) + proposal_log_density, dim=-1))
                          + self.SSM.observation_model.score(state=state, observation=observation, **data))
                return state, weight, resampled_weight, dynamic_log_density, proposal_log_density

        if isinstance(self.resampler, ConditionalResampler):
            def mpf_sampler(prev_state, prev_weight, observation, **data):
                state, weight, resampled_weight, dynamic_log_density, proposal_log_density = prop(prev_state, prev_weight, observation, **data)
                weight = torch.where(self.resampler.cache['mask'], weight, resampled_weight + torch.diagonal(dynamic_log_density, dim1=1, dim2=2) - torch.diagonal(proposal_log_density, dim1=1, dim2=2))
                try:
                    weight, likelihood = normalise(weight)
                except ValueError:
                    raise DivergenceError('Found batch where all weights are small.')

                return state, weight, torch.where(self.resampler.cache['mask'], likelihood, likelihood - normalise(resampled_weight)[1])
        else:
            def mpf_sampler(prev_state, prev_weight, observation, **data):
                state, weight, _, _, _ = prop(prev_state, prev_weight, observation, **data)
                try:
                    weight, likelihood = normalise(weight)
                except ValueError:
                    raise DivergenceError('Found batch where all weights are small.')
                return state, weight, likelihood

        super()._register_functions(initial_sampler, mpf_sampler)



class DPF(ParticleFilter):
    """
    Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi and O. Brock
    'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.
    """

    def __init__(self, SSM: FilteringModel = None,  resampling_generator: torch.Generator = torch.default_generator, multinomial:bool = False) -> None:
        """
        Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi, O. Brock
        'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(MultinomialResampler(resampling_generator), SSM, False)
        else:
            super().__init__(SystematicResampler(resampling_generator), SSM, False)

        temp = copy(self.proposal)
        self.proposal = lambda prev_state, prev_weight, **data: temp(prev_state.detach(), prev_weight.detach(), **data)

class MarginalDPF(MarginalParticleFilter):
    """
        Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi and O. Brock
        'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.
        """

    def __init__(self, SSM: FilteringModel = None, resampling_generator: torch.Generator = torch.default_generator, multinomial: bool = False) -> None:
        """
        Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi, O. Brock
        'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(MultinomialResampler(resampling_generator), SSM, 'none')
        else:
            super().__init__(SystematicResampler(resampling_generator), SSM, 'none')

        temp = copy(self.proposal)
        self.proposal = lambda prev_state, prev_weight, **data: temp(prev_state.detach(), prev_weight.detach(), **data)

class StraightThroughDPF(ParticleFilter):
    """
        Similar to the DPF but the gradient of the state is passed through resampling without modification. (T. Le et al. 'Auto-encoding sequential monte carlo' 2018,
        C. Maddison et al. ' Filtering variational objectives' 2018, and C. Naesseth et al. 'Variational sequential monte carlo' 2018)
    """

    def __init__(self, SSM: FilteringModel = None,  resampling_generator: torch.Generator = torch.default_generator, multinomial:bool = False) -> None:
        """
        Similar to the DPF but the gradient of the state is passed through resampling without modification. (T. Le et al. 'Auto-encoding sequential monte carlo' 2018,
        C. Maddison et al. ' Filtering variational objectives' 2018, and C. Naesseth et al. 'Variational sequential monte carlo' 2018)



        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(MultinomialResampler(resampling_generator), SSM, False)
        else:
            super().__init__(SystematicResampler(resampling_generator), SSM, False)


class MarginalStraightThroughDPF(MarginalParticleFilter):
    """
        Similar to the DPF but the gradient of the state is passed through resampling without modification. (T. Le et al. 'Auto-encoding sequential monte carlo' 2018,
        C. Maddison et al. ' Filtering variational objectives' 2018, and C. Naesseth et al. 'Variational sequential monte carlo' 2018)
    """

    def __init__(self, SSM: FilteringModel = None,  resampling_generator: torch.Generator = torch.default_generator, multinomial:bool = False) -> None:
        """
        Similar to the DPF but the gradient of the state is passed through resampling without modification. (T. Le et al. 'Auto-encoding sequential monte carlo' 2018,
        C. Maddison et al. ' Filtering variational objectives' 2018, and C. Naesseth et al. 'Variational sequential monte carlo' 2018)



        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(MultinomialResampler(resampling_generator), SSM, 'none')
        else:
            super().__init__(SystematicResampler(resampling_generator), SSM, 'none')


class SoftDPF(ParticleFilter):
    """
    Differentiable particle filter with soft-resampling (P. Karkus, D. Hsu and W. S. Lee
    'Particle Filter Networks with Application to Visual Localization' 2018).
    """

    def __init__(self, SSM: FilteringModel = None,
                 softness: float = 0.7,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial: bool = False) -> None:
        """
        Differentiable particle filter with soft-resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        softness: float
            The trade-off parameter between a uniform and the usual resampling distribution.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(SoftResampler(softness, MultinomialResampler(resampling_generator), resampling_generator.device), SSM)
            return
        super().__init__(SoftResampler(softness, SystematicResampler(resampling_generator), resampling_generator.device), SSM)

class MarginalSoftDPF(MarginalParticleFilter):
    """
        Differentiable particle filter with soft-resampling (P. Karkus, D. Hsu and W. S. Lee
        'Particle Filter Networks with Application to Visual Localization' 2018).
        """

    def __init__(self, SSM: FilteringModel = None,
                 softness: float = 0.7,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial: bool = False) -> None:
        """
        Differentiable marginal particle filter with soft-resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        softness: float
            The trade-off parameter between a uniform and the usual resampling distribution.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(SoftResampler(softness, MultinomialResampler(resampling_generator), resampling_generator.device), SSM)
            return
        super().__init__(SoftResampler(softness, SystematicResampler(resampling_generator), resampling_generator.device), SSM)

class OptimalTransportDPF(ParticleFilter):
    """
    Differentiable particle filter with optimal transport resampling (A. Corenflos, J. Thornton, G. Deligiannidis and A. Doucet
    'Differentiable Particle Filtering via Entropy-Regularized Optimal Transport' 2021).
    """
    def __init__(self, SSM: FilteringModel = None,
                 regularisation: float = 0.99,
                 step_size: float = 0.9,
                 min_update_size: float = 0.01,
                 max_iterations: int = 100,
                 transport_gradient_clip: float = 1.,
                 ) -> None:
        """
        Differentiable particle filter with optimal transport resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler from the proposal kernel, takes the state, weights and data and returns the new states and weights.
        regularisation: float
            The maximum strength of the entropy regularisation, in our implementation regularisation automatically chosen per sample and
             annealed.
        step_size: float
            The factor by which to decrease the entropy regularisation per Sinkhorn loop.
        min_update_size: float
            The size of update to the transport potentials below which iteration should stop.
        max_iterations: int
            The maximum number iterations of the Sinkhorn loop, before stopping. Regardless of convergence.
        transport_gradient_clip: float
            The maximum per-element gradient of the transport matrix that should be passed. Higher valued gradients will be clipped to this value.
        """
        super().__init__(OptimalTransportResampler(regularisation, step_size, min_update_size, max_iterations, transport_gradient_clip), SSM)

class StopGradientDPF(ParticleFilter):
    """
    Differentiable particle filter with stop-gradient resampling (A. Scibor and F. Wood
    'Differentiable Particle Filtering without Modifying the Forward Pass' 2021).
    """
    def __init__(self, SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial = False,
                 use_REINFORCE_for_proposal:bool = False) -> None:
        """
        Differentiable particle filter with stop-gradient resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.
        """
        if multinomial:
            super().__init__(StopGradientResampler(MultinomialResampler(resampling_generator)), SSM, use_REINFORCE_for_proposal)
            return
        super().__init__(StopGradientResampler(SystematicResampler(resampling_generator)), SSM, use_REINFORCE_for_proposal)

class MarginalStopGradientDPF(MarginalParticleFilter):
    def __init__(self, SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial = False) -> None:
        """
        Differentiable particle filter with marginalised stop-gradient resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.

        Warnings
        ---------
        In the current implementation, this filter is the only case where taking an SSM with a null proposal is not equivalent to the using the same module for both the proposal
        and dynamic models. This is because a partially reparameterised estimator exists for the bootstrap case but not with a generic proposal. If SSM has a non-null proposal
        then this algorithm is exactly equivalent to the REINFORCEDPF.
        """
        if multinomial:
            super().__init__(MultinomialResampler(resampling_generator), SSM, 'partial')
            return
        super().__init__(SystematicResampler(resampling_generator), SSM, 'partial')

class REINFORCEDPF(MarginalParticleFilter):
    def __init__(self, SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial = False) -> None:
        """
        Differentiable particle filter with marginalised stop-gradient resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        resampling_generator:
            The generator to track the resampling rng.

        Warnings
        ---------
        In the current implementation, this filter is the only case where taking an SSM with a null proposal is not equivalent to the bootstrap formulation.
        """
        if multinomial:
            super().__init__(MultinomialResampler(resampling_generator), SSM, 'full')
            return
        super().__init__(SystematicResampler(resampling_generator), SSM, 'full')


class KernelDPF(ParticleFilter):

    def __init__(self, SSM: FilteringModel = None, kernel: KernelMixture = None, use_REINFORCE_for_proposal:bool = False) -> None:
        """
            Differentiable particle filter with mixture kernel resampling (Younis and Sudderth 'Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes' 2024).



            Parameters
            ----------
            SSM: FilteringModel
                A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
                If this parameter is not None then the values of initial_proposal and proposal are ignored.
            initial_proposal: ImportanceSampler
                Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
                and returns the importance sampled state and weights
            proposal: ImportanceKernel
                Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
            kernel: KernelMixture
                The kernel mixture to convolve over the particles to form the KDE sampling distribution.

            Returns
            -------
            kernel_resampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
                A Module whose forward method implements kernel resampling.
        """
        if kernel is None:
            raise ValueError('Must specify a kernel mixture')

        super().__init__(KernelResampler(kernel), SSM, use_REINFORCE_for_proposal)


class VariationalDPF(ParticleFilter):

    def __init__(self, SSM: FilteringModel = None) -> None:
        """
            Differentiable particle filter with mixture kernel resampling (Younis and Sudderth 'Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes' 2024).



            Parameters
            ----------
            SSM: FilteringModel
                A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
                If this parameter is not None then the values of initial_proposal and proposal are ignored.
            initial_proposal: ImportanceSampler
                Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
                and returns the importance sampled state and weights
            proposal: ImportanceKernel
                Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
            kernel: KernelMixture
                The kernel mixture to convolve over the particles to form the KDE sampling distribution.

            Returns
            -------
            kernel_resampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
                A Module whose forward method implements kernel resampling.
        """

        super().__init__(VariationalResampler(), SSM)


class SVGDKernelDPF(ParticleFilter):

    def __init__(self, SSM: FilteringModel = None, kernel: KernelMixture = None, lr:float=1e-2, alpha:float=0.9, iterations:int=100, *, initial_proposal: ImportanceSampler = None, proposal: ImportanceKernel = None) -> None:
        """
            Differentiable particle filter with mixture kernel resampling (Younis and Sudderth 'Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes' 2024).



            Parameters
            ----------
            SSM: FilteringModel
                A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
                If this parameter is not None then the values of initial_proposal and proposal are ignored.
            initial_proposal: ImportanceSampler
                Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
                and returns the importance sampled state and weights
            proposal: ImportanceKernel
                Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
            kernel: KernelMixture
                The kernel mixture to convolve over the particles to form the KDE sampling distribution.

            Returns
            -------
            kernel_resampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
                A Module whose forward method implements kernel resampling.
        """
        if kernel is None:
            raise ValueError('Must specify a kernel mixture')

        super().__init__(SVGD_kernel_resampling(kernel, lr, alpha, iterations), SSM)

class AuxiliaryDPF(ParticleFilter):
    def __init__(self, SSM: FilteringModel, n_repeats, resampling_generator, multinomial:bool =False) -> None:
        if multinomial:
            super().__init__(AuxiliaryResampler(SSM, n_repeats, MultinomialResampler(resampling_generator)), SSM)
            return
        super().__init__(AuxiliaryResampler(SSM, n_repeats, SystematicResampler(resampling_generator)), SSM)