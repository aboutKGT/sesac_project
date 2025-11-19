class Node:
    def __init__(self, data) -> None:
        self.data = data 
        self.next: Node|None = None

class LinkedList:
    def __init__(self) -> None:
        self.head: Node|None = None
    def __len__(self):
        current = self.head
        cnt = 0
        while current:
            current = current.next
            cnt += 1
        return cnt
    # insert  
    def insert(self, index, data):
        if index < 0 or index > len(self):
            raise IndexError("Index out of range")
        
        new_node = Node(data)
        
        # 맨 앞에 삽입
        if index == 0:
            new_node.next = self.head
            self.head = new_node
            return
        
        # 중간이나 끝에 삽입
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
    def display(self):
        current: Node|None = self.head
        result = []
        while current:
            result.append(current.data)
            current = current.next
        print(result)
    def delete(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        
        if index == 0:
            self.head = self.head.next
            return
        
        current = self.head
        for _ in range(index - 1):
            current = current.next
        current.next = current.next.next

