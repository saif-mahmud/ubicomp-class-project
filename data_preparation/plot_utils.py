import matplotlib.pyplot as plt
from matplotlib import animation


# Ref: https://github.com/AssemblyAI-Examples/mediapipe-python/blob/main/nb_helpers.py

def time_animate(data, pose_connections, rotate_data=True, rotate_animation=False):
    frame_data = data[:, :, 0]

    figure = plt.figure()
    figure.set_size_inches(5, 5, True)
    ax = figure.add_subplot(projection='3d')

    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if rotate_data:
        plot = [ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:red')]

        for i in pose_connections:
            plot.append(ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                                  [-frame_data[2, i[0]], -frame_data[2, i[1]]],
                                  [-frame_data[1, i[0]], -frame_data[1, i[1]]],
                                  color='k', lw=1)[0])

        ax.view_init(elev=10, azim=120)

    else:
        ax.scatter(frame_data[0, :], frame_data[1, :], frame_data[2, :], color='tab:red')

        for i in pose_connections:
            ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                      [frame_data[1, i[0]], frame_data[1, i[1]]],
                      [frame_data[2, i[0]], frame_data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)

    def init():
        return figure,

    def animate(i):
        frame_data = data[:, :, i]

        for idxx in range(len(plot)):
            plot[idxx].remove()

        plot[0] = ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:red')

        idx = 1
        for pse in pose_connections:
            plot[idx] = ax.plot3D([frame_data[0, pse[0]], frame_data[0, pse[1]]],
                                  [-frame_data[2, pse[0]], -frame_data[2, pse[1]]],
                                  [-frame_data[1, pse[0]], -frame_data[1, pse[1]]],
                                  color='k', lw=1)[0]
            idx += 1

        if rotate_animation:
            ax.view_init(elev=10., azim=120 + (360 / data.shape[-1]) * i)

        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init, frames=data.shape[2], interval=10, blit=True)

    plt.xticks(rotation=90)
    plt.yticks(rotation=90)

    plt.grid()
    plt.close()

    return anim
