from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('task', None, 'Task to perform: extract, train, predict')

def main(argv):
    print(f'Task: {FLAGS.task}')

if __name__ == '__main__':
    flags.mark_flag_as_required('task')
    app.run(main)
