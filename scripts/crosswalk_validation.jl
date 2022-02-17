using AdversarialCrosswalkAD
using Turing
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

@model function crosswalk_model(mdp::AdversarialCrosswalkMDP, s0, var, x, N)
    if x === missing
        #x ~ filldist(Normal(0.0, var), N)
        x ~ filldist(MvNormal(zeros(6), var), N)
    end

    s = s0
    dist_traj = Vector{Real}(undef, N)
    for i = 1:N
        sp = step!(mdp, s, x[:, i])

        # Calculate distance
        x_diff = sp[1] - sp[3]
        y_diff = sp[4]
        d = sqrt(x_diff^2 + y_diff^2)
        push!(dist_traj, d)
    end

    #error = clamp(pi/2 - maximum(abs.(traj)), 0, Inf)
    temp = 0.001
    error = clamp(2 - minimum(dist_traj), 0, Inf)
    Turing.@addlogprob! logpdf(Exponential(temp), error)
end


v_des = 8. # m/s
sut_policy = IntelligentDriverModel(v_des=v_des)
mdp = AdversarialCrosswalkMDP(sut_policy, 0.1, 1.0, 4.0, 4.0)

s_ego = [-25., v_des]
s_ped = [0.0, -4.0, 0.0, 102]
s0 = vcat(s_ego, s_ped)

horizon = 50
var = Vector{Float64}([0.1, 0.1, 0.1, 0.1, 0.01, 0.1])
model = crosswalk_model(mdp, s0, var, missing, horizon)
# chain = sample(model, NUTS(100, 0.4), iterations)