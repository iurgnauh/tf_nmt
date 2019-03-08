import sys
import tensorflow as tf

def avg(ckpt_list):
    with tf.Session() as sess:
        var_list = []
        var_lists_from_models = []
        for ckpt in ckpt_list:
            f = open('checkpoint','w')
            print >>f,'model_checkpoint_path: "model.ckpt-'+ckpt+'"'
            f.close()
            var_list_wrt_model = []
            for var_name, _ in tf.contrib.framework.list_variables('./'):
                #print var_name
                var = tf.contrib.framework.load_variable('./', var_name)
                var_list_wrt_model.append((var,var_name))
            var_lists_from_models.append(var_list_wrt_model)
        for i in range(len(var_lists_from_models[0])):
            var_name = var_lists_from_models[0][i][1]
            if var_name == 'global_step':
                var_value = var_lists_from_models[0][i][0]
            else:
                var_values = []
                for j in range(len(var_lists_from_models)):
                    var_values.append(var_lists_from_models[j][i][0])
                var_value = sum(var_values)/len(var_values)
            var = tf.Variable(var_value, name=var_name)
            var_list.append(var)

        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.variables_initializer(var_list))
        saver.save(sess, './model.ckpt')

li = sys.argv[1:]
avg(li)

