
def flooding(loss, b):
    """flooding loss
    """
    return (loss - b).abs() + b
