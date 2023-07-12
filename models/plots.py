import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import	roc_curve, auc, confusion_matrix



def get_bins(x, label):
	q25, q75 = np.percentile(x, [25, 75])
	bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
	bins = round((x.max() - x.min()) / bin_width)
	#print("Freedman–Diaconis number of bins in " + label + ":", bins)

	return bins


def vad_values_statistics(dir_, x, y):
	valence = y[:, 0]
	arousal = y[:, 1]
	dominance = y[:, 2]

	plt.clf()
	fig, (axs1, axs2, axs3) = plt.subplots(1, 3)
	axs1.hist(valence, bins=get_bins(valence, 'valence'))
	axs1.set(xlabel='Valence', ylabel='Number of words')
	axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
	axs1.set_ylim([0, 1300])

	axs2.hist(arousal, bins=get_bins(arousal, 'arousal'))
	axs2.set(xlabel='Arousal')
	axs2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
	axs2.set_ylim([0, 1300])

	axs3.hist(dominance, bins=get_bins(dominance, 'dominance'))
	axs3.set(xlabel='Dominance')
	axs3.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
	axs3.set_ylim([0, 1300])

	plt.savefig(dir_ + 'figures/vad_sampling.png')

	#plt.show()
	#exit()



def create_save_plots(dir_name, r):
	plt.clf()
	plt.plot(r.history['loss'], label='loss')
	plt.plot(r.history['val_loss'], label='val_loss')
	plt.plot(r.history['output_reg_vad_loss'], label='output_reg_vad_loss')
	plt.plot(r.history['val_output_reg_vad_loss'], label='val_output_reg_vad_loss')
	plt.title("Regression mse, root_square_error")
	plt.legend()
	plt.savefig(dir_name + 'figures/vad_reg_mse_rse.png')
	#plt.show()
	plt.clf()

	plt.plot(r.history['loss'], label='loss')
	plt.plot(r.history['val_loss'], label='val_loss')
	plt.plot(r.history['output_class_pos_neg_loss'], label='output_class_pos_neg_loss')
	plt.plot(r.history['val_output_class_pos_neg_loss'], label='val_output_class_pos_neg_loss')
	plt.title("Loss classification")
	plt.legend()
	plt.savefig(dir_name + 'figures/loss_classification.png')
	#plt.show()
	plt.clf()

	plt.plot(r.history['output_class_pos_neg_acc_pos_neg'], label='output_class_pos_neg_acc_pos_neg')
	plt.plot(r.history['val_output_class_pos_neg_acc_pos_neg'], label='val_output_class_pos_neg_acc_pos_neg')
	plt.title("Accuracy classification")
	plt.legend()
	plt.savefig(dir_name + 'figures/acc_classification.png')
	#plt.show()
	plt.clf()



def create_save_confusion_matrix(dir_name, y_dev, y_pred, label, classes):
	y_test = [np.argmax(y, axis=0) for y in y_dev]
	y_pred = [np.argmax(y, axis=0) for y in y_pred]

	cf_matrix = confusion_matrix(y_test, y_pred)
	plt.figure(figsize=(5, 5))
	sns.heatmap(cf_matrix, annot=True, fmt="d")
	plt.title("Confusion matrix " + label)# (non-normalized)
	plt.ylabel("Actual label")
	plt.xlabel("Predicted label")
	plt.savefig(dir_name + 'figures/confusion_matrix_' + label + '.png')
	#plt.show()
	plt.clf()

	cf_matrix = np.array(cf_matrix)
	print(cf_matrix)



def create_save_roc_curve(dir_name, y_dev, y_pred, label, classes):
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	n_classes = np.shape(y_dev)[1]
	print(n_classes)
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_dev[:, i], y_pred[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Plot of a ROC curve for a specific class
	plt.figure()
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label='ROC curve %s(area = %0.2f)' % (classes[i], roc_auc[i]))
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curves ' + label)
	plt.legend(loc="lower right")
	plt.savefig(dir_name + 'figures/roc_curve_' + label + '.png')
	#plt.show()
	plt.clf()
