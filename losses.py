import tensorflow as tf

from keras import backend as K


# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/helper_functions.py#L37-L42
def dice_coef(y_true, y_pred):
    # Calculate dice coefficient
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py#L11-L53
def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        # Calculate binary focal loss
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss
    return binary_focal_loss_fixed

# modified code from https://stackoverflow.com/questions/53303724/how-to-average-only-non-zero-entries-in-tensor
# and https://stackoverflow.com/questions/42606207/keras-custom-decision-threshold-for-precision-and-recall
def volume_loss(threshold=0.25):
    def calculate(y_true, y_pred):
        # Calculate volume loss
        nonzero_true = K.any(K.not_equal(y_true, 0), axis=-1)
        y_true_volume = K.sum(K.cast(nonzero_true, 'float32'), axis=-1, keepdims=True)
        y_true_volume = K.sum(y_true_volume/1000)

        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
        nonzero_pred = K.any(K.not_equal(y_pred, 0), axis=-1)
        y_pred_volume = K.sum(K.cast(nonzero_pred, 'float32'), axis=-1, keepdims=True)
        y_pred_volume = K.sum(y_pred_volume/1000)
        
        return K.abs(y_true_volume-y_pred_volume)
    return calculate

def total_loss(type_loss='FL', gamma=2., alpha=.25, threshold=0.25):
    def calculate(y_true, y_pred):
        # Calculate total loss
        if type_loss == 'FTL':
            loss = focal_tversky(gamma=gamma, alpha=alpha) (y_true, y_pred)
        else:
            loss = binary_focal_loss(gamma=gamma, alpha=alpha) (y_true, y_pred)
        vl = volume_loss(threshold=threshold) (y_true, y_pred)
        
        return loss+vl
    return calculate