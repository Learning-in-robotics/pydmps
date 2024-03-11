import matplotlib.pyplot as plt

def plot_pose(y_tracks, raw_y_tracks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot(y_tracks[:, 0], y_tracks[:, 1], y_tracks[:, 2], label="DMP")
    ax.plot(raw_y_tracks[:, 0], raw_y_tracks[:, 1], raw_y_tracks[:, 2], label="Raw")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

def plot_2d(y_tracks, raw_y_tracks=None):
    fig, ax = plt.subplots(3, 2, figsize=(6, 6))
    ax[0, 0].plot(y_tracks[:, 0], y_tracks[:, 1])
    ax[0, 0].set_title("xy")
    ax[1, 0].plot(y_tracks[:, 1], y_tracks[:, 2])
    ax[1, 0].set_title("yz")
    ax[2, 0].plot(y_tracks[:, 2], y_tracks[:, 0])
    ax[2, 0].set_title("zx")

    if raw_y_tracks is not None:
        # plot trajectory_1
        ax[0, 1].plot(raw_y_tracks[:, 0], raw_y_tracks[:, 1])
        ax[0, 1].set_title("xy")
        ax[1, 1].plot(raw_y_tracks[:, 1], raw_y_tracks[:, 2])
        ax[1, 1].set_title("yz")
        ax[2, 1].plot(raw_y_tracks[:, 2], raw_y_tracks[:, 0])
        ax[2, 1].set_title("zx")

    plt.tight_layout()
    plt.show()