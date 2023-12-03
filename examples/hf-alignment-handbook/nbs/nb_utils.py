def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        else:
            return False
    except NameError:
        return False


IN_NOTEBOOK = in_notebook()

if IN_NOTEBOOK:
    import os, sys, nbdev
    from dotenv import load_dotenv

    sys.path.append("../")
    # load_dotenv("../api/.env")
