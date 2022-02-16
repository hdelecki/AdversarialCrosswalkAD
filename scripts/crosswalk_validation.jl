using AdversarialCrosswalkAD
using Turing
using Zygote
using Distributions
using LinearAlgebra


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

@model function crosswalk_model(mdp::AdversarialCrosswalkMDP, )


end

v_des = 11. # m/s
sut_policy = IntelligentDriverModel(v_des=v_des)

mdp = AdversarialCrosswalkMDP(sut_policy, 0.1, 1.0, 3.0, 1.0)

s_ego = [-20., v_des]
s_ped = [0.0, 0.0, 0.0, 0.2]
#s_ped = [0.0, -4.0, 0.0, 0.0]
s0 = vcat(s_ego, s_ped)

n_iter = 50
x = zeros(6, n_iter)

traj = simulate(mdp, x, s0)