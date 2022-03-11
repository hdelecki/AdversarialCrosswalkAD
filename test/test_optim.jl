using AdversarialCrosswalkAD
using Turing
using Zygote
using Optim
using Distributions
using LinearAlgebra
using DataFrames
using DelimitedFiles
using Plots

Turing.setadbackend(:zygote)

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

@model function crosswalk_model(mdp::AdversarialCrosswalkMDP, s0, var, x, N)
    if x === missing
        x ~ filldist(MvNormal(zeros(6), var), N)
    end

    s = s0
    dist_buffer = Zygote.Buffer(zeros(N))
    for i = 1:N
        sp = step!(mdp, s, x[:, i])
        # Calculate distance
        s = sp
        x_diff = s[1] - s[3]
        y_diff = s[4]
        d = sqrt(x_diff^2 + y_diff^2)
        dist_buffer[i] = d
    end

    dist_traj = copy(dist_buffer)
    temp = 0.001
    error = clamp(minimum(dist_traj) - 0.5, 0, Inf)
    #if error > 0
    Turing.@addlogprob! logpdf(Exponential(temp), error)
    #end
end

v_des = 11. # m/s
sut_policy = AdversarialCrosswalkAD.IntelligentDriverModel(v_des=v_des)
mdp = AdversarialCrosswalkMDP(sut_policy, 0.1, 1.0, 4.0, 3.0)

s_ego = [-25., v_des]
s_ped = [0.0, -4.0, 0.0, 1.4]
s0 = vcat(s_ego, s_ped)

horizon = 50
var = Vector{Float64}([0.1, 0.1, 0.1, 0.1, 0.01, 0.1])
var = sqrt.(var)
#std = Vector{Float64}([0.1, 0.1, 0.1, 0.1, 0.01, 0.1])
#var = std.^2
#model = crosswalk_model(mdp, s0, var, missing, horizon)
model = crosswalk_model(mdp, s0, var, missing, horizon)

mle = optimize(model, MLE())
xmle = reshape(mle.values.array, 6, :)
traj = simulate(mdp, xmle, s0)
plot(traj[3, :], traj[4, :])


# iterations = 100
# # params = zeros(6, horizon)
# #params[5, :] .= -1.0
# chain = sample(model, NUTS(iterations, 0.4), iterations, discard_adapt = false)


# df = DataFrame(chain)
# chain_data = Matrix(df)

# nsamples = size(chain_data, 1)
# @show nsamples

# sampled_trajectories = zeros(6, horizon, nsamples)
# min_dists = zeros(nsamples)
# for i=1:size(chain_data, 1)
#     samp = reshape(chain_data[i, 3:302], 6, :)
#     sampled_trajectories[:, :, i] = simulate(mdp, samp, s0)

#     min_dists[i] = minimum([ego_ped_distance(sampled_trajectories[:, j, i]) for j=1:horizon])
# end
# @show sum(min_dists .< 0.5)
# #@show df


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