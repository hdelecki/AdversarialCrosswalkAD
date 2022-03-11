using AdversarialCrosswalkAD
using CrossEntropyMethod
using Distributions
using Random
using LinearAlgebra
using DataFrames
using Plots

function simulate(mdp, x, s0)
    s = s0
    horizon = size(x, 2)
    trajectory_record = zeros(length(s), horizon)
    for i=1:horizon
        trajectory_record[:, i] = s
        sp = step!(mdp, s, x[:, i])
        s = sp
    end
    trajectory_record
end

function ego_ped_distance(s::Vector{Float64})
    x_diff = s[1] - s[3]
    y_diff = s[4]
    d = sqrt(x_diff^2 + y_diff^2)
    return d
end

function continous_actions_weight(d_true)
    function weight(d, s)
        #@show s
        w = exp(logpdf(d_true, s) - logpdf(d, s))
        return w
    end
end


function mdp_loss(mdp, failure, N)
    function loss(distribution, sample)
        s = s0
        traj = Vector{Float64}(undef, N)
        x = sample[:x]
        for i = 1:N
            traj[i] = failure(mdp, s)
            sp = step!(mdp, s, x[:, i])
            s = sp
        end
        f = minimum(traj)
        return f
    end
end

function mdp_loss_distance(mdp, failure_distance, s0, N)
    function loss(distribution, sample)
        s = s0
        traj = Vector{Float64}(undef, N)
        #x = sample[:x]
        #x = [sample[:x1],sample[:x2],sample[:x3],sample[:x4],sample[:x5],sample[:x6]]
        for i = 1:N
            traj[i] = failure_distance(mdp, s)
            #sp = step!(mdp, s, x[:, i])
            xi = [sample[:x1][1],
                sample[:x2][1],
                sample[:x3][1],
                sample[:x4][1],
                sample[:x5][1],
                sample[:x6][1]
            ]
            sp = step!(mdp, s, xi)
            s = sp
        end
        #d = -1.0*clamp(maximum(abs.(traj)), 0, pi/2)/(pi/2)
        # failure = -1
        # no failure = 0
        d = minimum(traj)
        return d
    end
end

function my_cross_entropy_method(loss,
                              d_in;
                              max_iter,
                              N=100,
                              elite_thresh = -0.99,
                              min_elite_samples = Int64(floor(0.1*N)),
                              max_elite_samples = typemax(Int64),
                              weight_fn = (d,x) -> 1.,
                              rng::AbstractRNG = Random.GLOBAL_RNG,
                              verbose = false,
                              show_progress = false,
                              batched = false,
                              add_entropy = (x)->x
                             )
    d = deepcopy(d_in)
    show_progress ? progress = Progress(max_iter) : nothing
    failure_samples = []

    for iteration in 1:max_iter
        # Get samples -> Nxm
        samples = rand(rng, d, N)

        # sort the samples by loss and select elite number
        if batched
            losses = loss(d,samples)
            @assert length(losses) == N
        else
            losses = [loss(d, s) for s in samples]    
        end
        f_samples = samples[losses .== -1]
        if !isempty(f_samples)
            push!(failure_samples, f_samples)
        end
        
        order = sortperm(losses)
        losses = losses[order]
        N_elite = losses[end] < elite_thresh ? N : findfirst(losses .> elite_thresh) - 1
        N_elite = min(max(N_elite, min_elite_samples), max_elite_samples)

        verbose && println("iteration ", iteration, " of ", max_iter, " N_elite: ", N_elite)

        # Update based on elite samples
        elite_samples = samples[order[1:N_elite]]
        weights = [weight_fn(d, s) for s in elite_samples]
        if all(weights .≈ 0.)
            println("Warning: all weights are zero")
        end
        d = fit(d, elite_samples, weights, add_entropy = add_entropy)
        show_progress && next!(progress)
    end
    return d, failure_samples
end

function is_failure(trajectory::Matrix{Float64}, threshold::Float64)
    for i=1:size(trajectory, 2)
        if ego_ped_distance(trajectory[:, i]) <= threshold
            return true
        end
    end
    return false
end


# MDP
horizon = 30
v_des = 11. # m/s
# s_ego = [-25., v_des]
# s_ped = [0.0, -5.0, 0.0, 1.4]
s_ego = [-15., v_des]
s_ped = [0.0, -2., 0.0, 1.4]
s0 = vcat(s_ego, s_ped)
sut_policy = AdversarialCrosswalkAD.IntelligentDriverModel(v_des=v_des)
mdp = AdversarialCrosswalkMDP(sut_policy, 0.1, 1.0, 4.0, 3.0)

# Disturbance Model
var = Vector{Float64}([0.1, 0.1, 0.1, 0.1, 0.01, 0.1])
var = sqrt.(var)


# true_disturbance_model = Dict{Symbol, Tuple{Sampleable, Int64}}(:x => (MvNormal(zeros(6), var), horizon))
# init_var = 4.0 .* var
# init_disturbance_model = Dict{Symbol, Tuple{Sampleable, Int64}}(:x => (MvNormal(zeros(6), init_var), horizon))
true_disturbance_model = Dict{Symbol, Tuple{Sampleable, Int64}}(
    :x1 => (Normal(0.0, var[1]), horizon),
    :x2 => (Normal(0.0, var[2]), horizon),
    :x3 => (Normal(0.0, var[3]), horizon),
    :x4 => (Normal(0.0, var[4]), horizon),
    :x5 => (Normal(0.0, var[5]), horizon),
    :x6 => (Normal(0.0, var[5]), horizon)
)

scale = 3.0
init_disturbance_model = Dict{Symbol, Tuple{Sampleable, Int64}}(
    :x1 => (Normal(0.0, scale*var[1]), horizon),
    :x2 => (Normal(0.0, scale*var[2]), horizon),
    :x3 => (Normal(0.0, scale*var[3]), horizon),
    :x4 => (Normal(0.0, scale*var[4]), horizon),
    :x5 => (Normal(0.0, scale*var[5]), horizon),
    :x6 => (Normal(0.0, scale*var[5]), horizon)
)


#failure_dist(mdp, s) = -1.0 + maximum([ego_ped_distance(s)-0.5, 0])/50.0
failure_dist(mdp, s) = ego_ped_distance(s) < 0.5 ? -1.0 : 0.0
loss_function = mdp_loss_distance(mdp, failure_dist, s0, horizon)
weight_function = continous_actions_weight(true_disturbance_model)

@time begin
    dopt, failure_samples = my_cross_entropy_method(loss_function,
                                        init_disturbance_model,
                                        weight_fn = weight_function,
                                        max_iter = 10,
                                        N = 10000,
                                        min_elite_samples = 10,
                                        verbose = true)
end

@show dopt
print(sum([length(l) for l in failure_samples]))


# xtest = transpose(hcat(
#         failure_samples[end][1][:x1],
#         failure_samples[end][1][:x2],
#         failure_samples[end][1][:x3],
#         failure_samples[end][1][:x4],
#         failure_samples[end][1][:x5],
#         failure_samples[end][1][:x6]
#     ))
# trajectory = simulate(mdp, xtest, s0)
# plot(trajectory[3, :], trajectory[4, :])
#all_samples = [failure_samples[i]... for i=1:length(failure_samples)]

# failure_lps = []
# for i=1:length(failure_samples)
#     for j=1:length(failure_samples[i])
#         push!(failure_lps, logpdf(true_disturbance_model, failure_samples[i][j]))
#     end
# end
n_failures = sum([length(failure_samples[i]) for i=1:length(failure_samples)])
cem_logp = []
cem_failure_vec = []
x_vec = []
for i=1:length(failure_samples)
    for j=1:length(failure_samples[i])
        #theta_trajectories[:, i] = simulate(mdp, policy, failure_samples[i][j][:x])
        
        xi= transpose(hcat(
            failure_samples[i][j][:x1],
            failure_samples[i][j][:x2],
            failure_samples[i][j][:x3],
            failure_samples[i][j][:x4],
            failure_samples[i][j][:x5],
            failure_samples[i][j][:x6]
        ))
        
        trajectory = simulate(mdp, xi, s0)
        
        #if is_failure(trajectory, 0.5)
        push!(cem_failure_vec, simulate(mdp, xi, s0))
        push!(cem_logp, logpdf(true_disturbance_model, failure_samples[i][j]))
        push!(x_vec, xi)

        #end
    end
end


lp_running_mean = [mean(cem_logp[1:i]) for i=1:length(cem_logp)]
plot(lp_running_mean)

lp_running_max = [maximum(cem_logp[1:i]) for i=1:length(cem_logp)]
plot(lp_running_max)








# Monte Carlo
# @time begin
#     mc_iterations = 1000000
#     mc_trajectories = zeros(6, horizon, mc_iterations)
#     mc_min_dists = zeros(mc_iterations)
#     xmc = filldist(MvNormal(zeros(6), var), horizon)
#     mc_t = zeros(mc_iterations)

#     for i = 1:mc_iterations
#         xs = rand(xmc)
#         mc_trajectories[:, :, i] = simulate(mdp, xs, s0)
#         mc_min_dists[i] = minimum([ego_ped_distance(mc_trajectories[:, j, i]) for j=1:horizon])
#     end
#     @show sum(mc_min_dists .< 0.5)
# end


# CEM





# p1 = plot()
# p2 = plot()
# p3 = plot()
# for i = 1:iterations
#     if min_dists[i] < 0.5
#         failure_trajectory = sampled_trajectories[:, :, i]
#         x_ped = failure_trajectory[3, :]
#         y_ped = failure_trajectory[4, :]
#         p1 = plot!(p1, x_ped, y_ped, color=:black, marker=:circle, legend=false)
        

#         samp = reshape(chain_data[i, 3:302], 6, :)
#         p2 = plot!(p2, x_ped.+samp[1, :], y_ped .+ samp[2, :], color=:black, marker=:circle, legend=false)
        
#         ovx_ped = samp[3, :]
#         ovy_ped = samp[4, :]
#         p3 = plot!(p3, ovx_ped, ovy_ped)
#     end
# end
# p1 = plot!(p1, axis_ratio=:equal)
# display(p1)

# p2 = plot!(p2, axis_ratio=:equal)
# display(p2)

# p3 = plot!(p3, axis_ratio=:equal)
# display(p3)