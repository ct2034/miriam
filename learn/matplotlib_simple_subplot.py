from matplotlib import pyplot as plt

test_x = [0, 1, 2]
test_accuracy = [.3, .7, .9]
test_loss = [.3, .2, .05]

fig, axs = plt.subplots(2)
axs[0].plot(test_x, test_accuracy, label="test_accuracy")
axs[1].plot(test_x, test_loss, label="test_loss")
axs[0].legend(loc='lower left')
axs[0].set_xlabel('Batch')
axs[1].legend(loc='lower left')
axs[1].set_xlabel('Batch')
fig.savefig("training_history.png")
fig.show()
