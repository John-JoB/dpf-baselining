from typing import Tuple
import pydpf
import torch
from torch import Tensor
from math import log2, log

class ParallelSmoother(pydpf.Module):
    def __init__(self, proposal, SSM):
        super().__init__()
        self.proposal = proposal
        self.SSM = SSM

    class GradientStabLog(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.log(input)

        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.saved_tensors[0]
            return torch.where(torch.eq(input, 0), torch.zeros_like(grad_output), grad_output/input)

    def logeinsumexp(self, equation, *operands, batch_dims=0):
        max = []
        new_operands = []
        for i, o in enumerate(operands):
            max.append(torch.max(o.flatten(start_dim=batch_dims), dim=-1)[0])
            new_operands.append(torch.exp(o - pydpf.match_dims(max[-1], o)))
        output = torch.einsum(equation, *new_operands)
        #Trajectories with very low weight contribute negligibly to sum so it's fine to clip them.
        return self.GradientStabLog.apply(output) +  pydpf.match_dims(torch.sum(torch.stack(max, dim=-1), dim=-1), output)


    class GradientStabReWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, ratios):
            mask = ratios.isinf()
            exp = torch.exp(ratios)
            ctx.save_for_backward(input, ratios, mask, exp)
            return torch.where(mask, 0, torch.clip(input * exp, -50, 50))

        @staticmethod
        def backward(ctx, grad_output):
            input, ratios, mask, exp = ctx.saved_tensors
            y = input*exp
            grad_mask = torch.logical_or(torch.logical_or(mask,  y > 50), y <-5 )
            return torch.where(grad_mask, 0, exp*grad_output), torch.where(grad_mask, 0, y*grad_output)


    def stabeinsumexp(self, equation, *operands, norm=torch.tensor(0), batch_dims=0, no_exp=None):
        if no_exp is None:
            no_exp = []
        new_operands = []
        max = []
        for i, o in enumerate(operands):
            if torch.isnan(o).any():
                #print(f'nan: {i}')
                pass
            if i in no_exp:
                new_operands.append(o)
                continue
            max.append(torch.max(o.flatten(start_dim=batch_dims), dim=-1)[0])
            new_operands.append(torch.exp(o - pydpf.match_dims(max[-1], o)))
        #print(torch.sum(torch.stack(max, dim=-1), dim=-1))
        #print(torch.stack(max, dim=-1).flatten(start_dim=0, end_dim=-2)[torch.argmax(torch.sum(torch.stack(max, dim=-1), dim=-1))])
        output = torch.einsum(equation, *new_operands)
        try:
            maxes = pydpf.match_dims(torch.sum(torch.stack(max, dim=-1), dim=-1).unsqueeze(1), output)
            #print('stabs')
            #print(maxes.flatten()[0])
            #print(norm[0,0].flatten())
            #print(torch.min(norm))
            diff = maxes-norm
            #print('maxes')
            #print(torch.max(output))
            #print(torch.max(diff))
            #print(torch.max(torch.clip(output * torch.exp(diff), -1e2, 1e2)))
            return self.GradientStabReWeight.apply(output, diff)
        except Exception as e:
            maxes = pydpf.match_dims(torch.sum(torch.stack(max, dim=-1), dim=-1), output)
            return output * torch.exp(maxes-norm)


    def combine(self, l, forward_kernels, proposals, conditional_likelihoods, depth, function = None, current_function = None):
        t = 1<<depth
        c_list = [not (((c/t) - 1) % 2) for c in range(proposals.size(0))]
        c_kernels = forward_kernels[c_list[1:]]
        c_likelihoods = conditional_likelihoods[c_list]
        c_proposals = proposals[c_list]
        c_factors = c_likelihoods - c_proposals
        if function is None:
            if depth == 0:
                return c_kernels + c_factors.unsqueeze(-2), None
            left_l = l[::2]
            right_l = l[1::2]
            return self.logeinsumexp('tbij,tbkl,tbjk,tbk->tbil', left_l, right_l, c_kernels, c_factors, batch_dims=2) - (2*log(forward_kernels.size(-1))), None
        else:
            if depth == 0:
                left_fun = function[::2]
                right_fun = function[1::2]
                left_fun = left_fun.unsqueeze(-2).expand(-1, -1, -1, left_fun.size(-2), -1)
                right_fun = right_fun.unsqueeze(-3).expand(-1, -1, right_fun.size(-2), -1, -1)
                return c_kernels + c_factors.unsqueeze(-2), torch.stack((left_fun, right_fun), dim=1).view(2*left_fun.size(0), left_fun.size(1), left_fun.size(2), left_fun.size(3), -1)
            b_list = [not (((b+1) / t) % 2) for b in range(proposals.size(0))]
            left_l = l[::2]
            right_l = l[1::2]
            new_l = self.logeinsumexp('tbij,tbkl,tbjk,tbk->tbil', left_l, right_l, c_kernels, c_factors, batch_dims=2)
            b_likelihoods = conditional_likelihoods[b_list]
            b_proposals = proposals[b_list]
            normalising_factor = new_l + (b_proposals - b_likelihoods).unsqueeze(-2)
            right_l_t = right_l + (b_proposals - b_likelihoods).unsqueeze(-2)
            reshaped_fun = current_function.reshape(-1, t, function.size(1), function.size(2), function.size(2), function.size(3))
            left_fun = reshaped_fun[::2]
            right_fun = reshaped_fun[1::2]
            #test = self.stabeinsumexp('tbij,tbkl,tbjk,tbk->tbil', left_l, right_l_t, c_kernels, c_factors, batch_dims=2, norm=normalising_factor)
            #print(test[:,0, 0, 0])
            new_right_fun = self.stabeinsumexp('tbij,tbkl,tbjk,tbk,thbkld->thbild', left_l, right_l_t, c_kernels, c_factors, right_fun, batch_dims=2, no_exp=[4], norm=normalising_factor.unsqueeze(1).unsqueeze(-1))
            new_left_fun = self.stabeinsumexp('tbij,tbkl,tbjk,tbk,thbijd->thbild', left_l, right_l_t, c_kernels, c_factors, left_fun, batch_dims=2, no_exp=[4], norm=normalising_factor.unsqueeze(1).unsqueeze(-1))
            return new_l - (2*log(forward_kernels.size(-1))), torch.stack((new_left_fun, new_right_fun), dim=1).view(2*left_fun.size(0),left_fun.size(1), left_fun.size(2), left_fun.size(3), left_fun.size(4), -1).flatten(0,1)





    def forward(self, n_particles: int, time_extent: int, observation, *, fun = 'L', ground_truth = None, control = None, time = None, series_metadata = None) -> Tuple[Tensor, Tensor]:
        assert  (((time_extent+1) & (time_extent)) == 0)
        particles, densities = self.proposal(n_particles, observation[:time_extent+1])
        particles_now = particles[1:].unsqueeze(2).expand(-1, -1, particles.size(2), -1, -1).flatten(start_dim=2,end_dim=3)
        particles_before = particles[:-1].unsqueeze(3).expand(-1, -1, -1, particles.size(2), -1).flatten(start_dim=2,end_dim=3)
        forward_kernels = torch.empty((particles.size(0)-1, particles.size(1), particles.size(2), particles.size(2)), device = particles_now.device)
        time_data = pydpf.SIS._get_time_data(0, observation=observation, ground_truth=ground_truth, control=control, series_metadata=series_metadata, time=time)
        conditional_likelihoods = torch.empty((particles.size(0), particles.size(1), particles.size(2)), device = particles_now.device)
        conditional_likelihoods[0] = self.SSM.observation_model.score(state = particles[0], **time_data)
        prior_density = self.SSM.prior_model.log_density(state=particles[0], **time_data)
        function = None
        if not fun == 'L':
            function = fun(particles)
        for t in range(time_extent):
           time_data = pydpf.SIS._get_time_data(t, observation=observation, ground_truth=ground_truth, control=control, series_metadata=series_metadata, time=time)
           forward_kernels[t] = self.SSM.dynamic_model.log_density(state=particles_now[t], prev_state=particles_before[t], **time_data).reshape(particles.size(1), particles.size(2), particles.size(2))
           conditional_likelihoods[t+1] = self.SSM.observation_model.score(state = particles[t+1], **time_data)
        l=None
        '''
        if not self.training:
            print('kernels')
            #print(forward_kernels[14, 0])
            print(torch.median(forward_kernels))
            print(torch.max(forward_kernels))
            print('cond likelihoods')
            print(torch.median(conditional_likelihoods))
            print(torch.max(conditional_likelihoods))
            print('prop density')
            print(torch.median(densities))
            print(torch.max(densities))
            print('prior density')
            print(torch.median(prior_density))
            print(torch.max(prior_density))
            '''
        #print(31*torch.mean(conditional_likelihoods[1:].unsqueeze(-2) + forward_kernels - densities[1:].unsqueeze(-2)))
        psi = None
        for depth in range(int(log2(time_extent))+1):
            l, psi = self.combine(l, forward_kernels, densities, conditional_likelihoods, depth, function, psi)
        l_t = l.squeeze(0) + (conditional_likelihoods[0] - densities[0] + prior_density).unsqueeze(-1)
        L = torch.logsumexp(l_t, dim=(1,2))
        if psi is not None:
            psi = torch.einsum('tbijd,bij->tbd', psi, torch.exp(l_t - pydpf.match_dims(L, l_t)))
            #print(torch.sum(torch.eq(psi, 0))/len(psi.flatten()))
        return L-2*log(n_particles), psi

class FCNN(pydpf.Module):
    def __init__(self, *layers, device):
        super().__init__()
        instantiated = []
        for i in range(len(layers)-1):
            instantiated.append(torch.nn.Linear(layers[i], layers[i+1], device=device))
            if i < len(layers)-2:
                instantiated.append(torch.nn.Tanh())
        self.seq = torch.nn.Sequential(*instantiated)

    def forward(self, x):
        return self.seq(x)


class SimpleProposal(pydpf.Module):
    def __init__(self, observation_dim, state_dim, generator):
        super().__init__()
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.mean_layer = FCNN(self.observation_dim, self.state_dim, self.state_dim, device=generator.device)
        self.cov_layer = FCNN(self.observation_dim, self.state_dim**2, self.state_dim**2, device=generator.device)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(state_dim, device=generator.device), torch.eye(state_dim, device=generator.device), False, generator=generator)

    def forward(self, n_particles, observations: Tensor):
        means = self.mean_layer(observations)
        temp = self.cov_layer(observations)
        temp = temp.reshape(temp.size(0), temp.size(1), self.state_dim, self.state_dim)
        temp = torch.tril(temp)
        temp[:,:, torch.arange(self.state_dim), torch.arange(self.state_dim)] = torch.abs(temp[:,:, torch.arange(self.state_dim), torch.arange(self.state_dim)])
        cov = temp
        sample = self.dist.sample((observations.size(0), observations.size(1), n_particles))
        #print(temp[0, 0])
        dets = torch.linalg.slogdet(temp)[1]
       # print(torch.mean(-(self.dist.log_density(sample)), dim=-1))
        return means.unsqueeze(-2) + torch.einsum('tbpi,tbij->tbpj', sample, cov), self.dist.log_density(sample) - dets.unsqueeze(-1)


