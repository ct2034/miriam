
class A():
    def __init__(self):
        self.foo = 0

    def print_a_foo(self, a=None):
        if a is None:
            a = self.foo
        print(a)


if __name__ == "__main__":
    a = A()
    a.print_a_foo(9)
    a.print_a_foo()
