using AdversarialCrosswalkAD


sut_policy = IntelligentDriverModel()
mdp = AdversarialCrosswalkMDP(sut_policy, 1.0, 1.0, 10.0, 10.0)

s_ego = []
s_ped = []
s = vcat(s_ego, s_ped)
x = zeros(6)