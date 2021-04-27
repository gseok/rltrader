class Logger:
  def __init__(self, message=None):
    self.message = message

  def log(self):
    print(self.message)

  def get_message(self):
    if self.message is not None:
      return self.message
    return None
