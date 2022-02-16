using AdversarialCrosswalkAD
using Plots

function simulate(mdp, x, s0)
    s = s0
    trajectory_record = zeros(length(s), n_iter)
    for i=1:n_iter
        trajectory_record[:, i] = s
        sp = step!(mdp, s, x[:, i])
        s = sp
    end
    trajectory_record
end

v_des = 8. # m/s
sut_policy = IntelligentDriverModel(v_des=v_des)

mdp = AdversarialCrosswalkMDP(sut_policy, 0.1, 1.0, 4.0, 3.0)

s_ego = [-25., 8.]
#s_ped = [0.0, 0.0, 0.0, 0.2]
s_ped = [0.0, -4, 0.0, 1.0]
s0 = vcat(s_ego, s_ped)

n_iter = 50
x = zeros(6, n_iter)

traj = simulate(mdp, x, s0)

p1 = plot(traj[1, :], zeros(n_iter), marker=:circle)
p1 = plot!(p1, traj[3, :], traj[4, :], marker=:circle)
p1 = plot!(axis_ratio = :equal)
display(p1)


ped_dist = sqrt.((traj[1, :] .- traj[3, :]).^2 + traj[4, :].^2)
d = plot(ped_dist)
display(d) 

traj[1, :]
