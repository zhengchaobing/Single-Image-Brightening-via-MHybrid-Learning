from network import *
from PIL import Image
import scipy.misc as misc
import os
from pylab import *
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',    dest='phase',   default='test', help='train or test')
parser.add_argument('--save_dir', dest = 'save_dir', default='H:\Experiments\Paper_3\Results_images\Result_CNN_CRF\CNN_mid', help='directory for testing outputs')
parser.add_argument('--test_dir',  dest = 'test_dir',  default='H:\Experiments\Paper_3\Results_images\Result_CNN_CRF\V_mid',  help='directory for testing inputs')

# parser.add_argument('--save_dir', dest = 'save_dir', default='H:\Experiments\Paper_3\Results_images\Result_CNN_CRF\CNN_hig', help='directory for testing outputs')
# parser.add_argument('--test_dir',  dest = 'test_dir',  default='H:\Experiments\Paper_3\Results_images\Result_CNN_CRF\V_hig',  help='directory for testing inputs')



args = parser.parse_args()


def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate(   [result_1, result_2], axis = 1)

    im = Image.fromarray(np.uint8(np.clip(cat_image * 255.0, 0, 255.0)))
    im.save(filepath)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0


def VDSR_Coefficient_tf(input_im, param):
    mask1 = tf.where(condition=tf.less(input_im, param), x=tf.zeros_like(input_im), y=tf.ones_like(input_im))
    mask2 = tf.where(condition=tf.greater_equal(input_im, param), x=tf.zeros_like(input_im), y=1 / (((param+1/255.) - input_im)*255) )
    coefficient = tf.add(mask1, mask2, name="coefficient")
    return coefficient


def cos_loss(target, prediction, name=None):
    with tf.name_scope(name, default_name='cos_loss', values=[target, prediction]):
        norm = tf.linalg.norm(target, axis=-1) * tf.linalg.norm(prediction, axis=-1)
        dot = tf.reduce_sum(tf.multiply(target, prediction), axis=-1)
        loss = tf.reduce_mean( 1-dot / (norm + 1e-5))
    return loss


def cos_loss1(target, prediction, name=None):
    with tf.name_scope(name, default_name='cos_loss', values=[target, prediction]):
        loss = tf.abs(tf.losses.cosine_distance(target, prediction, axis=0))
    return loss


def tv_loss(input_, output):
    I = tf.image.rgb_to_grayscale(input_)
    L = I + 0.00001
    dx = L[:, :-1, :-1, :] - L[:, :-1, 1:, :]
    dy = L[:, :-1, :-1, :] - L[:, 1:, :-1, :]

    alpha = tf.constant(1.2)
    lamda = tf.constant(1.5)
    dx = tf.divide(lamda, tf.pow(tf.abs(dx), alpha) + tf.constant(0.0001))
    dy = tf.divide(lamda, tf.pow(tf.abs(dy), alpha) + tf.constant(0.0001))

    shape = output.get_shape()
    x_loss = dx * ((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) ** 2)
    y_loss = dy * ((output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) ** 2)
    tvloss = tf.reduce_mean(x_loss + y_loss) / 2.0
    return tvloss



class DnCNN:

    def __init__(self):
        self.clean_img   = tf.placeholder(tf.float32,  [None, None, None, IMG_C])
        self.noised_img  = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)
        dncnn = net("DnCNN")

        self.denoised_img  = dncnn(self.noised_img, self.train_phase)

        coefficient = VDSR_Coefficient_tf(self.noised_img ,noise_lever / 255.)

        self.losses_r = tf.reduce_mean( tf.reduce_sum(coefficient*tf.square((self.denoised_img - self.clean_img)), axis=0))
        self.losses_c = cos_loss(self.clean_img, self.denoised_img)
        self.losses_s = tv_loss(self.clean_img, self.denoised_img)
        self.loss = W_r * self.losses_r + W_c * self.losses_c + W_smooth * self.losses_s

        self.Opt = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())


    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        saver.save(self.sess,os.path.join(ckpt_dir, model_name),global_step=iter_num)


    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0


    def train(self):
        clean_filepath  = "./data2/Hig/"
        noise_filepath  = "./data2/V_Hig/"
        filenames = os.listdir(clean_filepath)
        saver = tf.train.Saver()

        train_loss = self.loss
        summaries = [
            tf.summary.scalar('loss/train_loss', train_loss),
            tf.summary.scalar('losses_r', self.losses_r),
            tf.summary.scalar('losses_c', self.losses_c),
            tf.summary.scalar('loss_smooth', self.losses_s),
        ]

        load_model_status, global_step = self.load(saver, "./save_para_H3")
        iter_num=0

        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // (filenames.__len__()//BATCH_SIZE)
            start_step = global_step % (filenames.__len__()//BATCH_SIZE)
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            global_step=0
            print("[*] Not find pretrained model!")

        write_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter("./train", tf.get_default_graph())

        for epoch in range(start_epoch, EPOCHS):
            for x in range(8):
                for y in range(8):
                    for batch_id in range((filenames.__len__()//BATCH_SIZE)):
                        cleaned_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C])
                        noised_batch  = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C])
                        for patch_id, filename in enumerate(filenames[batch_id*BATCH_SIZE:batch_id*BATCH_SIZE+BATCH_SIZE]):
                            cleaned_batch[patch_id, :, :, :] = (np.array(Image.open(clean_filepath + filename), dtype="float32") / 255.0)[y * IMG_H: (y + 1) * IMG_H, x * IMG_W: (x + 1) * IMG_W, :]
                            noised_batch [patch_id, :, :, :] = (np.array(Image.open(noise_filepath + filename),  dtype="float32") / 255.0)[y * IMG_H: (y + 1) * IMG_H, x * IMG_W: (x + 1) * IMG_W, :]

                        loss,_,summary=self.sess.run([[self.loss, self.losses_r, self.losses_c, self.losses_s], self.Opt, write_op], feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: True})
                        global_step += 1
                        writer_train.add_summary(summary, global_step)
                        writer_train.flush()

                        print("Epoch: [{}], [{} /{}], loss: {}, loss_r: {}, loss_c: {}, loss_smooth: {}".format(epoch + 1, batch_id + 1, (filenames.__len__()//BATCH_SIZE), loss[0], loss[1], loss[2], loss[3]))

                        iter_num+=1

                        if (epoch + 1) % eval_every_epoch == 0:
                            self.save(saver, iter_num, "./save_para_H3", "DnCNN")

    def test(self):
        saver = tf.train.Saver()

        load_model_status_DnCNN, global_step3 = self.load(saver, "./save_para_M2")

        if load_model_status_DnCNN:
            print("[*] Load weights successfully...")

        if args.test_dir == None:
            print("输入数据的地址为空")
            exit(0)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
        test_low_data = []
        for idx in range(len(test_low_data_name)):
            test_low_im = load_images(test_low_data_name[idx])  # 读取数据并转化为【0 1】之间
            test_low_data.append(test_low_im)  # 把图像数据放进列表


        for idx in range(len(test_low_data)):
            # 打印图像的路径
            print(test_low_data_name[idx])
            [_, name] = os.path.split(test_low_data_name[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            # 获得低照度图像的数据
            input_low_test = np.expand_dims(test_low_data[idx], axis=0)  # 就是在axis的那一个轴上把数据加上去
            print("测试的shape", input_low_test.shape)
            plt.figure(idx)
            plt.imshow(input_low_test[0])
            plt.title(str(idx))

            denoised_img= self.sess.run(self.denoised_img, feed_dict={self.noised_img: input_low_test, self.train_phase: False})

            print("S的shape", denoised_img.shape)
            plt.figure(idx + 5)
            plt.imshow(np.squeeze(denoised_img))
            print("S2的shape", np.squeeze(denoised_img).shape)
            plt.title('OutPut' + str(idx))
            save_images(os.path.join(args.save_dir, name + "." + "png"), denoised_img)
        plt.show()




if __name__ == "__main__":
    dncnn = DnCNN()
    if args.phase == 'train':
           dncnn.train()
    elif args.phase == 'test':
           dncnn.test()