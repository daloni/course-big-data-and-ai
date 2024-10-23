class Matrix:
  def __init__(self, values):
    self.values = values

  def __getitem__(self, index):
    return self.values[index]

  def __add__(self, other):
    if isinstance(other, Matrix):
      return Matrix(self.values + other.values)
    else:
      raise TypeError("param is not a Matrix type")

  def __mul__(self, other):
    if not isinstance(other, Matrix):
      raise TypeError("param is not a Matrix type")

    selfRows = len(self.values)
    selfCols = len(self.values[0])
    otherRows = len(other.values)
    otherCols = len(other.values[0])

    result = [[0] * otherCols for _ in range(selfRows)]

    if selfCols != otherRows:
      raise ValueError('Values are not compatible')

    for i in range(selfRows):
      for j in range(otherCols):
        for k in range(selfCols):
          result[i][j] += self.values[i][k] * other.values[k][j]

    return Matrix(result)


object1 = Matrix([[2, -3, -5], [-1, 4, 5], [1, -3, -4]])
object2 = Matrix([[2, 2, 0], [-1, -1, 0], [1, 2, 1]])

object3 = object1 * object2
print(object3.values)

object4 = object2 * object1
print(object4.values)
