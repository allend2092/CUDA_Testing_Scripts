# Import necessary libraries
import torch
import matplotlib.pyplot as plt

# Define the number of particles
N = 1000

# Generate random initial positions for the particles in a 2D space.
# The positions are stored in a tensor of shape (N, 2) and are moved to the GPU using `device='cuda:0'`.
positions = torch.rand((N, 2), device='cuda:0')

# Generate random initial velocities for the particles in the same 2D space.
# These velocities are scaled down by multiplying with 0.1 to ensure they are small.
# Like the positions, these velocities are also moved to the GPU.
velocities = torch.rand((N, 2), device='cuda:0') * 0.1

# Set up the plot for visualization
plt.ion()  # Turn on interactive mode for real-time updates
fig, ax = plt.subplots()

# Create a scatter plot using the initial positions of the particles.
# The positions are moved to CPU and converted to numpy arrays for plotting.
sc = ax.scatter(positions[:, 0].cpu().numpy(), positions[:, 1].cpu().numpy())

# Set the x and y axis limits for the plot. Adjust these values to change the scale of the graph.
ax.set_xlim(0, 15)  # Adjust the maximum value (e.g., 2) to increase the x-axis scale
ax.set_ylim(0, 15)  # Adjust the maximum value (e.g., 2) to increase the y-axis scale

# Set the title for the plot
ax.set_title("Particle Movement Simulation")

# Display the initial state of the plot
plt.draw()

# Simulate the movement of particles for 100 time steps.
for i in range(150):
    # Update the positions of the particles based on their velocities
    positions += velocities

    # Update the scatter plot with the new positions of the particles
    sc.set_offsets(positions.cpu().numpy())

    # Introduce a short delay between each frame to visualize the movement
    plt.pause(0.1)

# Turn off interactive mode and display the final state of the plot
plt.ioff()
plt.show()
