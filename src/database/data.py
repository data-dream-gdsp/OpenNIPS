class table:
    def __init__(self, key:list, value:list, quite:bool=True):

        if len(key) != len(value): raise RuntimeError("Wrong data")

        self.key = key
        self.value = value
        self.table = {}
        self.quite = quite

        cnt = 0
        while cnt <= len(key):
            table[key] = table[value]

    def __repr__(self) -> str:
        return str(self.table)

    def get(self, key:str) -> str:
        return str(self.table.get(key))
    
    def set(self, key:str, value:str) -> bool:
        try:
            self.table[key] = value
            return True
        except Exception as e:
            if not self.quite:
                raise RuntimeError(e)
            else:
                return False