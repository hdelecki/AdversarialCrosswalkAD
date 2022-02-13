module AdversarialCrosswalkAD
    using POMDPs
    using POMDPPolicies
    using POMDPSimulators
    using POMDPModelTools
    # using AutomotiveSimulator
    # using AutomotiveVisualization
    using Distributions
    using Parameters
    using Random

    export IntelligentDriverModel, action, set_desired_speed!, reset_hidden_state!
    include("intelligent_driver_model.jl")

    export AdversarialCrosswalkMDP, step
    include("mdp.jl")




end