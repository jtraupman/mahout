package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;

/**
 * Implements linear operators of the form (A + ve') where A is a linear operator, v is vector of size
 * equal to the range (number of rows) of A, and e is a vector of all ones equal in size to the
 * domain (number of columns) of A.
 * 
 * Operators of this form can be used to efficiently represent the common situation of subtracting a mean
 * vector from a sparse data matrix. The sparsity of the underlying data in the linear operator A is left
 * unchanged, so using this linear operator class will not impact algorithms' performance like explicitly
 * adding/subtracting an offset from the rows of A, which would make it dense.
 *
 */

public class ColumnOffsetLinearOperator extends AbstractLinearOperator {
  private LinearOperator linop;
  private Vector offset;
  
  /**
   * Constructor for the general case of a vector offset.
   * 
   * @param linop The linear operator to offset.
   * @param offset The vector to add to each row of linop.
   * @throws CardinalityException if the size of offset differs from the number of rows in linop.
   */
  public ColumnOffsetLinearOperator(LinearOperator linop, Vector offset) {
    if (offset.size() != linop.numRows()) {
      throw new CardinalityException(offset.size(), linop.numRows());
    }
    this.linop = linop;
    this.offset = offset;
  }

  /**
   * Constructor for the special case of an offset vector having all elements equal to the same value.
   * 
   * @param linop The linear operator to offset.
   * @param offset The value of each element in the offset vector.
   */
  public ColumnOffsetLinearOperator(LinearOperator linop, double offset) {
    this.linop = linop;
    this.offset = new DenseVector(linop.numRows());
    this.offset.assign(offset);
  }
  
  @Override
  public int numRows() {
    return linop.numRows();
  }
  
  @Override
  public int numCols() {
    return linop.numCols();
  }
  
  @Override
  public Vector times(Vector v) {
    Vector result = linop.times(v);
    double sumv = v.aggregate(Functions.PLUS, Functions.IDENTITY);
    result.assign(offset, Functions.plusMult(sumv));
    return result;
  }

  @Override
  public LinearOperator transpose() {
    return new RowOffsetLinearOperator(linop.transpose(), offset);
  }  
}
