import matplotlib.pyplot as plt

# Data points
sparsity = [80.69, 40.05, 80.69, 40.05]
accuracy = [92.850, 91.670, 92.750, 89.310]
labels = ['imp_unstructured', 'imp_filter', 'omp_unstructured', 'omp_filter']
colors = ['blue', 'red', 'green', 'orange']  # Assign unique colors to each point

# Plot the data
plt.figure(figsize=(8, 6))
for i in range(len(sparsity)):
    plt.scatter(sparsity[i], accuracy[i], label=labels[i], s=100, color=colors[i])  # Scatter points
    # Annotate each point with its name and sparsity, in the same color as the point
    plt.text(
        sparsity[i], accuracy[i] + 0.5, 
        f"{labels[i]} {sparsity[i]:.2f} {accuracy[i]}%", 
        ha='left', fontsize=9, color=colors[i]
    )

# Add labels and legend
plt.title("Accuracy vs. Sparsity")
plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.xlim(0, 100)  # X-axis range (Sparsity)
plt.ylim(88, 94)  # Y-axis range (Accuracy)
plt.grid(alpha=0.4)
plt.legend(loc="lower left")
plt.tight_layout()

# Show the plot
plt.show()
