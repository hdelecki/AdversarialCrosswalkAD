using CSV
using DataFrames
using Plots

df = CSV.read("./data/hmc_wallclock.csv", DataFrame)


p1 = plot(LinRange(0, 250, 99), df[!, "0_1"], label="HMC", linewidth=2, legend=:topleft)
p1 = plot!(p1, LinRange(0, 250, 99), zeros(99), label="MC", color=:black, linewidth=2)
xlabel!(p1,"Sampling Time (s)")
ylabel!(p1,"# Failures (-)")
savefig(p1, "./data/wallclock_plot.png")