import tkinter as tk


def login():
    class Toto:
        def __init__(self, root):
            self.var1 = tk.StringVar()
            self.root = root
            self.root.geometry('300x160')
            self.root.title('Enter your information')
            # frame for window margin
            self.parent = tk.Frame(self.root, padx=10, pady=10)
            self.parent.pack(fill=tk.BOTH, expand=True)
            # entrys with not shown text
            self.user = self.make_entry(self.parent, "User name:", 16, show='')
            self.password = self.make_entry(self.parent, "Password:", 16, show="*")
            # button to attempt to login
            self.button = tk.Button(self.parent, borderwidth=4, text="Login", width=10, pady=8,
                                    command=self.get_value)
            self.button.pack(side=tk.BOTTOM)
            self.user.focus_set()
            self.USR = None
            self.MDP = None
            self.root.bind('<Return>', self.push_enter)

        def make_entry(self, _, caption, width=None, **options):
            tk.Label(self.parent, text=caption).pack(side=tk.TOP)
            entry = tk.Entry(self.parent, **options)
            if width:
                entry.config(width=width)
            entry.pack(side=tk.TOP, padx=10, fill=tk.BOTH)
            return entry

        def push_enter(self, _):
            self.get_value()

        def get_value(self):
            self.USR = self.user.get()
            self.MDP = self.password.get()
            self.root.destroy()
            self.root.quit()

    root = tk.Tk()
    toto = Toto(root)
    root.mainloop()

    return toto.USR, toto.MDP
