def main():
    import random
    import matplotlib.pyplot as plt

    random.seed(7)
    room = [[random.choice([0, 1]) for _ in range(3)] for _ in range(3)]

    class StateAgent:
        def __init__(self):
            self.x = self.y = 0
            self.battery = 100
            self.state = "SEARCHING"
            self.score = 0

        def step(self):
            if self.state == "SEARCHING":
                if self.battery <= 20:
                    self.state = "RETURNING"
                elif room[self.x][self.y]:
                    self.state = "SUCKING"
                else:
                    self.x, self.y = random.randrange(3), random.randrange(3)
                    self.battery -= 5
            elif self.state == "SUCKING":
                room[self.x][self.y] = 0
                self.battery -= 10
                self.score += 10
                self.state = "SEARCHING"
            elif self.state == "RETURNING":
                self.x = self.y = 0
                self.battery -= 5
                self.state = "CHARGING"
            elif self.state == "CHARGING":
                self.battery = min(100, self.battery + 25)
                if self.battery == 100:
                    self.state = "SEARCHING"

    robot = StateAgent()
    for i in range(1, 9):
        if random.random() < 0.3:
            room[random.randrange(3)][random.randrange(3)] = 1
        robot.step()
        print(f"Step {i}: Position=({robot.x},{robot.y}), "
              f"State={robot.state}, Battery={robot.battery}%")
    print("Final Score:", robot.score)

    plt.imshow(room, cmap="RdYlGn_r", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, "Dirty" if room[i][j] else "Clean",
                     ha="center", va="center", fontsize=9)
    plt.scatter(robot.y, robot.x, marker="*", s=250, color="blue")
    plt.xticks(range(3))
    plt.yticks(range(3))
    plt.title("State-Based Agent - Final Room Status")
    plt.show()


if __name__ == "__main__":
    main()
