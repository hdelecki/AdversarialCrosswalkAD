using AdversarialCrosswalkAD

v_des = 11. # m/s
sut_policy = IntelligentDriverModel(v_des=v_des)

mdp = AdversarialCrosswalkMDP(sut_policy, 1.0, 1.0, 3.0, 1.0)

s_ego = [-10., v_des]
s_ped = [0.0, -2.0, 0.0, 1.4]
s = vcat(s_ego, s_ped)
x = zeros(6)

sp = step!(mdp, s, x)