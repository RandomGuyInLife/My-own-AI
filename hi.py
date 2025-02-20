import math

running = True

def check_int(s: str):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

while (running):
    input1 = input("Enter a^2 or exit: ")
    if input1 == "exit":
        running = False
        break
    elif check_int(input1):
        input2 = input("Enter b^2 or exit: ")
        if input2 == "exit":
            running = False
            break
        elif check_int(input2):
            print(math.sqrt(input1**2+input2**2))