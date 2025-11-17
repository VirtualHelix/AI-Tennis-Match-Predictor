class Node:
    def __init__(self, key, value):
        self.key = key        # match date
        self.value = value    # match row (pandas Series)
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key, value):
        if self.root is None:
            self.root = Node(key, value)
        else:
            self._insert_recursive(self.root, key, value)

    def _insert_recursive(self, curr, key, value):
        if key < curr.key:
            if curr.left is None:
                curr.left = Node(key, value)
            else:
                self._insert_recursive(curr.left, key, value)
        else:
            if curr.right is None:
                curr.right = Node(key, value)
            else:
                self._insert_recursive(curr.right, key, value)

    def search(self, key):
        return self._search_recursive(self.root, key)

    def _search_recursive(self, curr, key):
        if curr is None:
            return None
        if curr.key == key:
            return curr.value
        elif key < curr.key:
            return self._search_recursive(curr.left, key)
        else:
            return self._search_recursive(curr.right, key)

    def inorder(self):
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, curr, result):
        if curr:
            self._inorder_recursive(curr.left, result)
            result.append((curr.key, curr.value))
            self._inorder_recursive(curr.right, result)