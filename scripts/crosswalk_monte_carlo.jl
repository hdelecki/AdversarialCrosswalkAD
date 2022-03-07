using AdversarialCrosswalkAD
using Turing
using Zygote
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


v_des = 11. # m/s
sut_policy = IntelligentDriverModel(v_des=v_des)
mdp = AdversarialCrosswalkMDP(sut_policy, 0.1, 1.0, 4.0, 3.0)

s_ego = [-25., v_des]
s_ped = [0.0, -3.0, 0.0, 1.4]
s0 = vcat(s_ego, s_ped)

horizon = 50
var = Vector{Float64}([0.1, 0.1, 0.1, 0.1, 0.01, 0.1])


# Monte Carlo
@time begin
    mc_iterations = 5000000
    mc_trajectories = zeros(6, horizon, mc_iterations)
    mc_min_dists = zeros(mc_iterations)
    xmc = filldist(MvNormal(zeros(6), var), horizon)
    mc_t = zeros(mc_iterations)

    for i = 1:mc_iterations
        xs = rand(xmc)
        mc_trajectories[:, :, i] = simulate(mdp, xs, s0)
        mc_min_dists[i] = minimum([ego_ped_distance(mc_trajectories[:, j, i]) for j=1:horizon])
    end
    @show sum(mc_min_dists .< 0.5)
end


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