import numpy as np
import os

from Prostredie.EnvItem import *

class Prostredie:
    # Startovacie body
    startPositions_training = [(0,1), (9,1), (0,5)]
    startPositions_testing  = [(9,4), (6,8), (3,0)]

    # body v bludisku 
    prostredie = []
    
    def __init__(self, area_width, area_height, prostredie=None):
        self.area_width = area_width
        self.area_height = area_height
        self.currentPositionX = None
        self.currentPositionY = None
        
        # create log files
        self.f_startPosition = open("log_start.txt", "w")
        self.f_apples = open("log_apples.txt", "w")
        self.f_mines = open("log_mines.txt", "w")

        if (prostredie != None):
            for i in range(area_height * area_width):
                if (prostredie[i] == Jablko.Tag):
                    self.prostredie.append(Jablko())
                elif (prostredie[i] == Mina.Tag):
                    self.prostredie.append(Mina())
                elif (prostredie[i] == Cesta.Tag):
                    self.prostredie.append(Cesta())
                elif (prostredie[i] == Priepast.Tag):
                    self.prostredie.append(Priepast())
                elif (prostredie[i] == Vychod.Tag):
                    self.prostredie.append(Vychod())
        
    def GenerateItem(self, item):
        x = np.random.randint(1, self.area_width-1)
        y = np.random.randint(1, self.area_height-1)

        while (self.getId(x, y) != Cesta.Tag) or ((x == 3 and y == 0) or (x == 9 and y == 4) or (x == 6 and y == 8) or (x == 8 and y >= 5) or (y == 1) or (y == 5) or (x == 5 and y >= 5)):
            x = np.random.randint(1, self.area_width-1)
            y = np.random.randint(1, self.area_height-1)

        # prepis novou polozkou sveta
        self.prostredie[y*self.area_width + x] = item

        return x, y

    def Hodnotenie(self, x, y):
            if (self.getId(x, y) == Priepast.Tag):
                return -0.75     # zakazana zona
            elif (self.getId(x, y) == Jablko.Tag):
                return +0.20     # jablcko (odmena)
            elif (self.getId(x, y) == Mina.Tag):
                return -0.20     # mina (trest)
            elif (self.getId(x, y) == Vychod.Tag):
                return +1.00     # vychod z bludiska      
            else:
                return -0.015    # najkratsia cesta k vychodu

    def NahradObjekty(self, tag, item):
        for i in range(self.area_height * self.area_width):
            if (self.prostredie[i].id == tag):
                self.prostredie[i] = item
        
    def render(self):
            #os.system('clear')

            for i in range(self.area_width * 4 + 1):
                print('-', end='')
            print()

            for y in range(self.area_height):
                for x in range(self.area_width):
                    if (self.currentPositionX == x and self.currentPositionY == y):
                        # Vykresli agenta
                        print("|\033[0;103;30m A \033[0m", end='')
                    elif (abs(self.currentPositionX - x) <= 1 and abs(self.currentPositionY - y) <= 1):                       
                        print(f"|{self.prostredie[y*self.area_width + x]}", end='')
                    else:
                        print(f"| ", end='')

                print("|")
                for i in range(self.area_width * 4 + 1):
                    print('-', end='')
                print()

    def reset(self, testing=False):
        if testing:
            pos = self.startPositions_training[np.random.randint(0,len(self.startPositions_training))]
        else:
            pos = self.startPositions_training[np.random.randint(0,len(self.startPositions_training))]

        self.currentPositionX = pos[0]
        self.currentPositionY = pos[1]
        
        # Vymaz vsetky jablka a miny
        self.NahradObjekty(Jablko.Tag, Cesta())
        self.NahradObjekty(Mina.Tag, Cesta())

        # generator jablk
        self.count_apple = np.random.randint(3,5)
        for i in range(self.count_apple):
            x, y = self.GenerateItem(Jablko())
            self.f_apples.write(f"{x};{y}|")
        self.f_apples.write("\n")

        # generator min
        self.count_mine = np.random.randint(1,3)
        for i in range(self.count_mine):
            x, y = self.GenerateItem(Mina())
            self.f_mines.write(f"{x};{y}|")
        self.f_mines.write("\n")

        self.apples, self.mines = 0, 0

        state = [
            self.currentPositionX/10.0,
            self.currentPositionY/10.0
        ]
        state.extend(self.Radar())

        # log to file
        self.f_startPosition.write(f"{self.currentPositionX};{self.currentPositionY}\n")

        return np.array(state)

    def sample(self):
        akcia = np.random.randint(0, 4)
        return akcia

    def step(self, action):
        oldX = self.currentPositionX
        oldY = self.currentPositionY
        #print(self.currentPositionX, self.currentPositionY)

        if action == 0:     # dole
            self.currentPositionY += 1
        elif action == 1:   # hore
            self.currentPositionY -= 1
        elif action == 2:   # vlavo
            self.currentPositionX -= 1
        elif action == 3:   # vpravo
            self.currentPositionX += 1

        #print(action, self.currentPositionX, self.currentPositionY)

        nasiel_ciel = 0

        # skontroluj spravnost kroku v hre
        if (self.JeMimoAreny(self.currentPositionX, self.currentPositionY, 10, 10)):
            self.currentPositionX = oldX
            self.currentPositionY = oldY                
       
            reward = -1.00
            done = True   # koniec hry
        else:
            reward = self.Hodnotenie(self.currentPositionX, self.currentPositionY)

            # hra konci najdenim ciela
            if self.getId(self.currentPositionX, self.currentPositionY) == Vychod.Tag:
                done = True
                nasiel_ciel = 1
            else:
                done = False

        state = [
            self.currentPositionX/10.0,
            self.currentPositionY/10.0
        ]
        state.extend(self.Radar())


        # Agent zobral jablko
        if (self.getId(self.currentPositionX, self.currentPositionY) == Jablko.Tag):
            self.prostredie[self.currentPositionY*self.area_width + self.currentPositionX] = Cesta()
            self.apples += 1

        # Agent aktivoval minu
        if (self.getId(self.currentPositionX, self.currentPositionY) == Mina.Tag):
            self.prostredie[self.currentPositionY*self.area_width + self.currentPositionX] = Cesta()
            self.mines += 1

        return (np.array(state), reward, done, {'apples': self.apples, 'mines': self.mines, 'end': nasiel_ciel})
    
    def getId(self, x, y):
        return self.prostredie[y*self.area_width + x].id

    def upravaDatRadaru(self, x, y):
        if (self.getId(x, y) == Priepast.Tag):
            return [0, 0, 0]
        elif (self.getId(x, y) == Jablko.Tag):
            return [1, 0, 0]
        elif (self.getId(x, y) == Mina.Tag):
            return [0, 0, 1]
        elif (self.getId(x, y) == Cesta.Tag):
            return [0, 1, 0]
        elif (self.getId(x, y) == Vychod.Tag):
            return [1, 1, 1]

    def Radar(self):
        scan = []

        # Hore
        if self.JeMimoAreny(self.currentPositionX, self.currentPositionY-1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX, self.currentPositionY-1))
        else:
            scan.extend([0, 0, 0])

        # Vpravo-Hore
        if self.JeMimoAreny(self.currentPositionX+1, self.currentPositionY-1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX+1, self.currentPositionY-1))
        else:
            scan.extend([0, 0, 0])

        # Vpravo
        if self.JeMimoAreny(self.currentPositionX+1, self.currentPositionY, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX+1, self.currentPositionY))
        else:
            scan.extend([0, 0, 0])

        # Vpravo-Dole
        if self.JeMimoAreny(self.currentPositionX+1, self.currentPositionY+1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX+1, self.currentPositionY+1))
        else:
            scan.extend([0, 0, 0])

        # Dole
        if self.JeMimoAreny(self.currentPositionX, self.currentPositionY+1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX, self.currentPositionY+1))
        else:
            scan.extend([0, 0, 0])

        # Vlavo-Dole
        if self.JeMimoAreny(self.currentPositionX-1, self.currentPositionY+1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX-1, self.currentPositionY+1))
        else:
            scan.extend([0, 0, 0])

        # Vlavo
        if self.JeMimoAreny(self.currentPositionX-1, self.currentPositionY, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX-1, self.currentPositionY))
        else:
            scan.extend([0, 0, 0])

        # Vlavo-Hore
        if self.JeMimoAreny(self.currentPositionX-1, self.currentPositionY-1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX-1, self.currentPositionY-1))
        else:
            scan.extend([0, 0, 0])

        return scan

    def JeMimoAreny(self, x, y, sirka, vyska):
        if (x >= 0 and x < sirka and y >= 0 and y < vyska):
            return False

        return True
