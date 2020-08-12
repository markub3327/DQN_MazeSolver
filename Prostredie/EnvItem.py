class EnvItem:
    def __init__(self):
        self.text = ""
        self.id = 0

class Cesta(EnvItem):
    Tag = 1
    def __init__(self):
        self.text = "+"
        self.id = self.Tag

    def __str__(self):
        return f"\033[0;107;30m {self.text} \033[0m"

class Jablko(EnvItem):
    Tag = 2
    def __init__(self):
        self.text = "*"
        self.id = self.Tag
    
    def __str__(self):
        return f"\033[0;101;30m {self.text} \033[0m"

class Mina(EnvItem):
    Tag = 3
    def __init__(self):
        self.text = "M"
        self.id = self.Tag

    def __str__(self):
        return f"\033[0;106;30m {self.text} \033[0m"
    
class Priepast(EnvItem):
    Tag = 0
    def __init__(self):
        self.text = " "
        self.id = self.Tag

    def __str__(self):
        return f"\033[0m {self.text} "

class Vychod(EnvItem):
    Tag = 4
    def __init__(self):
        self.text = "E"
        self.id = self.Tag

    def __str__(self):
        return f"\033[0;102;30m {self.text} \033[0m"