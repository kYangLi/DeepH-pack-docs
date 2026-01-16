import time
from deepx._version import pkg_name, pkg_version

def info():
    _welcome_interface()
    
    time.sleep(0.5)
    
    messages = [
        ("\033[1;32m🚀 To use the DeepH JAX version (DeepX), apply for permission at:\033[0m", 0.05),
        ("\033[4;34m🔗 https://ticket.deeph-pack.com\033[0m", 0.4),
        ("", 0.2),
        
        ("\033[1;32m📚 Online Documentation:\033[0m", 0.05),
        ("\033[4;34m🔗 https://docs.deeph-pack.com/deeph-pack\033[0m", 0.4),
        ("", 0.2),
        

        
        ("\033[1;32m💡 DeepH-dock, a wonderful Data API for DeepH-pack:\033[0m", 0.05),
        ("\033[4;34m🔗 https://github.com/kYangLi/DeepH-dock\033[0m", 0.4),
        ("", 0.1),
        
        ("\033[1;32m📝 If you use `DeepH-pack` in your work, please cite:\033[0m", 0.05),
        ("", 0.1),
        
        ("\033[1;33m📄 The original framework paper introduced the foundational methodology:\033[0m", 0.05),
        ("    He Li, Zun Wang, Nianlong Zou, et al. Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation. Nat. Comput. Sci. 2, 367 (2022)", 0.05),
        ("\033[4;34m🔗 https://doi.org/10.1038/s43588-022-00265-6\033[0m", 0.05),
        ("", 0.5),
        
        ("\033[1;33m📄 Complete package featuring the latest implementation, methodology, and workflow:\033[0m", 0.05),
        ("    Yang Li, Yanzhen Wang, Boheng Zhao, et al. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)", 0.05),
        ("\033[4;34m🔗 https://arxiv.org/abs/2601.02938\033[0m", 0.05),
        ("", 0.5),
        
        ("\033[1;32m✨ Happy Computing! ✨\033[0m", 0.1)
    ]
    
    for text, delay in messages:
        if text:
            print(text)
        else:
            print()
        time.sleep(delay)


def _welcome_interface():
    welcome_lines = [
        r"                                                   ",
        f"⚡           Welcome to DeepH-pack ({pkg_name})!     ",
        f"⚡                 Version {pkg_version}             ",
        r"⚡                                                   ",
        r"⚡...................................................",
        r"⚡........_____....................._...._.[PACK]....",
        r"⚡.......|  __ \...................| |..| |..........",
        r"⚡.......| |  | | ___  ___ ._ _ ...| |..| |..........",
        r"⚡.......| |  | |/ _ \/ _ \| '_ \ .|X'><'X|..........",
        r"⚡.......| |__| |. __/. __/| |_) |.| |..| |..........",
        r"⚡.......|_____/ \___|\___|| .__/ .|_|..|_|..........",
        r"⚡.........................| |.......................",
        r"⚡.........................|_|.......................",
        r"⚡...................................................",
        r"⚡                                                   ",
        r"⚡            Copyright CMT@Phys.Tsinghua            ",
        r"⚡                 Powered by JAX                    ",
        r"                                                    ",
        r"                                                    "
    ]
    
    for line in welcome_lines:
        print(line)
        time.sleep(0.05)

