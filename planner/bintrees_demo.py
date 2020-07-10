import bintrees

if __name__ == "__main__":
    t = bintrees.BinaryTree()
    t.insert(8, "Hi")
    t.insert(-90, "Freaks!")
    print(t[8])
    for x in t:
        print(t[x])
        break
