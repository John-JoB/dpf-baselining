from typing import Tuple
import pydpf
import torch
from torch import Tensor
from math import log2, log
import einops

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
        big = False
        for i, o in enumerate(operands):
            #if torch.any(torch.isnan(o)):
            #    print(i)
            if o.dim() == 6:
                max.append(torch.max(o.detach().flatten(start_dim=batch_dims+1), dim=-1)[0])
                max[-1] = torch.where(torch.isinf(max[-1]), 0, max[-1])
                new_operands.append(torch.exp(o - pydpf.match_dims(max[-1], o)))
                big = True
                continue
            max.append(torch.max(o.detach().flatten(start_dim=batch_dims), dim=-1)[0])
            max[-1] = torch.where(torch.isinf(max[-1]), 0,  max[-1])
            new_operands.append(torch.exp(o - pydpf.match_dims(max[-1], o)))
            if big:
                max[-1] = max[-1].unsqueeze(1).expand_as(max[0])
        output = torch.einsum(equation, *new_operands)
        #Trajectories with very low weight contribute negligibly to sum so it's fine to clip them.
        max_factor = pydpf.match_dims(torch.sum(torch.stack(max, dim=-1), dim=-1), output)
        out = self.GradientStabLog.apply(output) + max_factor
        #test_max = torch.max(out.flatten(start_dim=batch_dims), dim=-1)[0]
        #if torch.isinf(test_max).any():
            #print(operands[1].size(0))
            #print(torch.min(max_factor))
        return out


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


    def combine(self, l, forward_kernels, depth, function = None, current_function = None):
        t = 1<<depth
        c_list = [not (((c/t) - 1) % 2) for c in range(forward_kernels.size(0)+1)][1:]
        c_kernels = forward_kernels[c_list]
        if function is None:
            if depth == 0:
                return c_kernels + log(c_kernels.size(-1)), None
            left_l = l[::2]
            right_l = l[1::2]
            with torch.profiler.record_function('Combine'):
                return self.logeinsumexp('tbij,tbkl,tbjk->tbil', left_l, right_l, c_kernels, batch_dims=2), None
        else:
            if depth == 0:
                stretched_fun = einops.repeat(function, 't b n d -> t b n m d', m=function.size(-2))
                left_fun = stretched_fun[::2]
                right_fun = stretched_fun[1::2]
                ls = c_kernels + log(c_kernels.size(-1))
                return ls, einops.rearrange([ls.unsqueeze(-1)+left_fun, ls.unsqueeze(-1)+right_fun], 'side t b n m d -> (t side) b n m d')
            left_l = l[::2]
            right_l = l[1::2]
            #reshaped_fun = einops.rearrange(current_function, '(t_new h) b n m d -> t_new h b n m d', t_new = function.size(0)//(t),  h = t)
            #left_fun = reshaped_fun[::2]
            #right_fun = reshaped_fun[1::2]
            #new_left_fun = self.logeinsumexp('thbijd,tbkl,tbjk->thbild', left_fun, right_l, c_kernels, batch_dims=2)
            #new_right_fun = self.logeinsumexp('thbkld,tbij,tbjk->thbild', right_fun, left_l, c_kernels, batch_dims=2)
            #new_l = self.logeinsumexp('tbij,tbkl,tbjk->tbil', left_l, right_l, c_kernels, batch_dims=2)
            #return  new_l,  einops.rearrange([new_left_fun, new_right_fun], 'side t h b n m d -> (t side h) b n m d')
    """
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
    """


    def finish(self, function, factors, forward_kernels, prior_density, time_extent,mins):
        psi = None
        l = None
        mod_kernels = forward_kernels + factors[1:].unsqueeze(-2)

        with torch.profiler.record_function('combination_step'):
            for depth in range(int(log2(time_extent)) + 1):
                l, psi = self.combine( l, mod_kernels, depth, function, psi)
                # l, psi = self.combine(l, mod_kernels, depth, function, psi)
            time_0_factors = (factors[0] + prior_density).unsqueeze(-1)
            # print(torch.max(l.flatten(start_dim=1), dim=-1))
            l_t = l.squeeze(0) + time_0_factors
            L = torch.logsumexp(l_t, dim=(1, 2))
            if psi is not None:
                psi = psi.squeeze(0) + time_0_factors.unsqueeze(-1)
                psi = torch.logsumexp(psi, dim=(2, 3))
                # psi = torch.einsum('tbijd,bij->tbd', psi, torch.exp(l_t - pydpf.match_dims(L, l_t)))
                # print(torch.sum(torch.eq(psi, 0))/len(psi.flatten()))
                return L, torch.exp(psi - L.unsqueeze(0).unsqueeze(-1)) - 1 + mins
        return L, None

    def forward(self, n_particles: int, time_extent: int, observation, *, fun = 'L', ground_truth = None, control = None, time = None, series_metadata = None) -> Tuple[Tensor, Tensor]:
        assert  (((time_extent+1) & (time_extent)) == 0)
        with torch.profiler.record_function('make_tensors'):
            particles, densities = self.proposal(n_particles, observation[:time_extent+1])
            particles_now = einops.repeat(particles[1:], 't b n d -> (t b) (m n) d', m=particles.size(-2))
            particles_before = einops.repeat(particles[:-1], 't b n d -> (t b) (n m) d', m=particles.size(-2))
            time_data = pydpf.SIS._get_time_data(0, observation=observation, ground_truth=ground_truth, control=control, series_metadata=series_metadata, time=time)
            prior_density = self.SSM.prior_model.log_density(state=particles[0], **time_data)
            function = None
            observation = einops.rearrange(observation, 't b d -> (t b) d')
            particles_squish = einops.rearrange(particles, 't b n d -> (t b) n d')
            conditional_likelihoods = einops.rearrange(self.SSM.observation_model.score(state=particles_squish, observation=observation), '(t b) n -> t b n', t=particles.size(0))
            forward_kernels = einops.rearrange(self.SSM.dynamic_model.log_density(state=particles_now, prev_state=particles_before), '(t b) (m n) -> t b m n', t = particles.size(0)-1, m = particles.size(-2))
            mins = None
            if not fun == 'L':
                t_fun = fun(particles)
                mins = torch.min(t_fun, dim=-2)[0]
                function = torch.log(t_fun + 1 - mins.unsqueeze(-2))
            factors = conditional_likelihoods - densities - (2 * log(forward_kernels.size(-1)))
        return self.finish(function, factors, forward_kernels, prior_density, time_extent, mins)
        #return torch.utils.checkpoint.checkpoint(self.finish, function, factors, forward_kernels, prior_density, time_extent, mins, use_reentrant=False)

        #print(31*torch.mean(conditional_likelihoods[1:].unsqueeze(-2) + forward_kernels - densities[1:].unsqueeze(-2)))

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


class ConvolutionalProposal(pydpf.Module):
    def __init__(self, observation_dim, state_dim, kernel_size, generator):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.trailing_time_steps = kernel_size - 1
        self.mean_layer_1 = torch.nn.Conv1d(in_channels=observation_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.mean_layer_2 = torch.nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.mean_network = torch.nn.Sequential(self.mean_layer_1, torch.nn.ReLU(), self.mean_layer_2)
        self.mean_start_network = FCNN(observation_dim*(self.trailing_time_steps*2-1), state_dim*self.trailing_time_steps, state_dim*self.trailing_time_steps, device=generator.device)
        self.mean_end_network = FCNN(observation_dim*(self.trailing_time_steps*2-1), state_dim*self.trailing_time_steps, state_dim*self.trailing_time_steps, device=generator.device)

        self.cov_layer_1 = torch.nn.Conv1d(in_channels=observation_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.cov_layer_2 = torch.nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.cov_network = torch.nn.Sequential(self.cov_layer_1, torch.nn.ReLU(), self.cov_layer_2)
        self.cov_start_network = FCNN(observation_dim * (self.trailing_time_steps*2-1), state_dim * self.trailing_time_steps, state_dim * self.trailing_time_steps, device=generator.device)
        self.cov_end_network = FCNN(observation_dim * (self.trailing_time_steps*2-1), state_dim * self.trailing_time_steps, state_dim * self.trailing_time_steps, device=generator.device)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(state_dim, device=generator.device), torch.eye(state_dim, device=generator.device), False, generator=generator)

    def forward(self, n_particles, observation):
        observation = torch.movedim(observation, 0, -1)
        means_main = self.mean_network(observation)
        means_start = self.mean_start_network(observation[:,:,:self.trailing_time_steps*2-1].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        means_end = self.mean_end_network(observation[:,:,-(self.trailing_time_steps*2-1):].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        cov_main = self.cov_network(observation)
        cov_start = self.cov_start_network(observation[:,:,:self.trailing_time_steps*2-1].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        cov_end = self.cov_end_network(observation[:,:,-(self.trailing_time_steps*2-1):].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        mean = torch.cat((means_start, means_main, means_end), dim=-1).movedim(-1,0).contiguous()
        cov =  torch.abs(torch.cat((cov_start, cov_main, cov_end), dim=-1).movedim(-1,0)).contiguous()
        sample = self.dist.sample((observation.size(-1), observation.size(0), n_particles))
        dets = torch.sum(torch.log(cov), dim=-1)
        return mean.unsqueeze(-2) + sample*cov.unsqueeze(-2), self.dist.log_density(sample) - dets.unsqueeze(-1)








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


