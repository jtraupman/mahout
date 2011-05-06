package org.apache.mahout.math.solver;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalOffsetLinearOperator;
import org.apache.mahout.math.LinearOperator;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SquaredLinearOperator;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class TestConjugateGradientSolver extends MahoutTestCase
{
  @Test
  public void testConjugateGradientSolver() {
    Matrix a = getA();
    Vector b = getB();
    
    ConjugateGradientSolver solver = new ConjugateGradientSolver();
    Vector x = solver.solve(a, b);
    
    assertEquals(0.0, Math.sqrt(a.times(x).getDistanceSquared(b)), EPSILON);    
    assertEquals(0.0, solver.getResidualNorm(), ConjugateGradientSolver.DEFAULT_MAX_ERROR);
    assertEquals(10, solver.getIterations());
  }
  
  @Test
  public void testConditionedConjugateGradientSolver() {
    Matrix a = getIllConditionedMatrix();
    Vector b = getB();
    Preconditioner conditioner = new JacobiConditioner(a);
    ConjugateGradientSolver solver = new ConjugateGradientSolver();
    
    Vector x = solver.solve(a, b, null, 100, ConjugateGradientSolver.DEFAULT_MAX_ERROR);

    double distance = Math.sqrt(a.times(x).getDistanceSquared(b));
    assertEquals(0.0, distance, EPSILON);
    assertEquals(0.0, solver.getResidualNorm(), ConjugateGradientSolver.DEFAULT_MAX_ERROR);
    assertEquals(16, solver.getIterations());
    
    Vector x2 = solver.solve(a, b, conditioner, 100, ConjugateGradientSolver.DEFAULT_MAX_ERROR);

    // the Jacobi preconditioner isn't very good, but it does result in one less iteration to converge
    distance = Math.sqrt(a.times(x2).getDistanceSquared(b));
    assertEquals(0.0, distance, EPSILON);
    assertEquals(0.0, solver.getResidualNorm(), ConjugateGradientSolver.DEFAULT_MAX_ERROR);
    assertEquals(15, solver.getIterations());
  }
  
  @Test
  public void testAsymmetricMatrix() {
    Matrix a = getAsymmetricMatrix();
    Vector b = getSmallB();    
    ConjugateGradientSolver solver = new ConjugateGradientSolver();
    
    try {
      solver.solve(a, b);
      assertTrue(false); // should not be able to build a symmetric solver with an asymmetric input
    } catch (IllegalArgumentException e) {
      assertTrue(true);
    }
    
    LinearOperator ata = new SquaredLinearOperator(a);
    
    Vector x = solver.solve(ata, b);
    double distance = Math.sqrt(ata.times(x).getDistanceSquared(b));
    assertEquals(0.0, distance, EPSILON);
    assertEquals(0.0, solver.getResidualNorm(), ConjugateGradientSolver.DEFAULT_MAX_ERROR);
    assertEquals(5, solver.getIterations());
  }
  
  @Test
  public void testSingularSymmetricMatrix() {
    Matrix a = getLowrankSymmetricMatrix();
    Vector b = getSmallB();
    ConjugateGradientSolver solver = new ConjugateGradientSolver();

    // solution will fail to converge because the matrix is singular
    Vector x = solver.solve(a, b);
    double distance = Math.sqrt(a.times(x).getDistanceSquared(b));
    assertTrue(distance > EPSILON);
    
    // adding a small constant to it will make it non-singular (c.f. ridge regression)

    LinearOperator offsetA = new DiagonalOffsetLinearOperator(a, 0.01);
    
    x = solver.solve(offsetA, b);
    distance = Math.sqrt(offsetA.times(x).getDistanceSquared(b));
    assertEquals(0.0, distance, EPSILON);
    assertEquals(0.0, solver.getResidualNorm(), ConjugateGradientSolver.DEFAULT_MAX_ERROR);
    assertEquals(3, solver.getIterations());
  }
  
  @Test
  public void testAsymmetricSingularMatrix() {
    Matrix a = getLowrankAsymmetricMatrix();
    Vector b = getSmallB();    
    ConjugateGradientSolver solver = new ConjugateGradientSolver();
    LinearOperator ata = new SquaredLinearOperator(a);
    
    Vector x = solver.solve(ata, b);
    double distance = Math.sqrt(a.timesSquared(x).getDistanceSquared(b));
    assertTrue(distance > EPSILON);

    LinearOperator offsetAtA = new DiagonalOffsetLinearOperator(ata, 0.01);
    
    x = solver.solve(offsetAtA, b);
    distance = Math.sqrt(offsetAtA.times(x).getDistanceSquared(b));
    assertEquals(0.0, distance, EPSILON);
    assertEquals(0.0, solver.getResidualNorm(), ConjugateGradientSolver.DEFAULT_MAX_ERROR);
    assertEquals(2, solver.getIterations());
  }
  
  @Test
  public void testEarlyStop() {
    Matrix a = getA();
    Vector b = getB();    
    ConjugateGradientSolver solver = new ConjugateGradientSolver();
    
    // specifying a looser max error will result in few iterations but less accurate results
    Vector x = solver.solve(a, b, null, 10, 0.1);
    double distance = Math.sqrt(a.times(x).getDistanceSquared(b));
    assertTrue(distance > EPSILON);
    assertEquals(0.0, distance, 0.1); // should be equal to within the error specified
    assertEquals(7, solver.getIterations()); // should have taken fewer iterations
    
    // can get a similar effect by bounding the number of iterations
    x = solver.solve(a, b, null, 7, ConjugateGradientSolver.DEFAULT_MAX_ERROR);    
    distance = Math.sqrt(a.times(x).getDistanceSquared(b));
    assertTrue(distance > EPSILON);
    assertEquals(0.0, distance, 0.1);
    assertEquals(7, solver.getIterations()); 
  }  
  
  private static Matrix getA() {
    return reshape(new double[] {
        11.7155649822793997, -0.7125253363083646, 4.6473613961860183,  1.6020939468348456, -4.6789817799137134,
        -0.8140416763434970, -4.5995617505618345, -1.1749070042775340, -1.6747995811678336, 3.1922255171058342,
        -0.7125253363083646, 12.3400579683994867, -2.6498099427000645, 0.5264507222630669, 0.3783428369189767,
        -2.1170186159188811, 2.3695134252190528, 3.8182131490333013, 6.5285942298270347, 2.8564814419366353,
         4.6473613961860183, -2.6498099427000645, 16.1317933921668484, -0.0409475448061225, 1.4805687075608227,
        -2.9958076484628950, -2.5288893025027264, -0.9614557539842487, -2.2974738351519077, -1.5516184284572598,
         1.6020939468348456, 0.5264507222630669, -0.0409475448061225, 4.1946802122694482, -2.5210038046912198,
         0.6634899962909317, 0.4036187419205338, -0.2829211393003727, -0.2283091172980954, 1.1253516563552464,
        -4.6789817799137134, 0.3783428369189767, 1.4805687075608227, -2.5210038046912198, 19.4307361862733430,
        -2.5200132222091787, 2.3748511971444510, 11.6426598443305522, -0.1508136510863874, 4.3471343888063512,
        -0.8140416763434970, -2.1170186159188811, -2.9958076484628950, 0.6634899962909317, -2.5200132222091787,
         7.6712334419700747, -3.8687773629502851, -3.0453418711591529, -0.1155580876143619, -2.4025459467422121,
        -4.5995617505618345, 2.3695134252190528, -2.5288893025027264, 0.4036187419205338, 2.3748511971444510,
        -3.8687773629502851, 10.4681666057470082, 1.6527180866171229, 2.9341795819365384, -2.1708176372763099,
        -1.1749070042775340, 3.8182131490333013, -0.9614557539842487, -0.2829211393003727, 11.6426598443305522,
        -3.0453418711591529, 1.6527180866171229, 16.0050616934176233, 1.1689747208793086, 1.6665090945954870,
        -1.6747995811678336, 6.5285942298270347, -2.2974738351519077, -0.2283091172980954, -0.1508136510863874,
        -0.1155580876143619, 2.9341795819365384, 1.1689747208793086, 6.4794329751637481, -1.9197339981871877,
         3.1922255171058342, 2.8564814419366353, -1.5516184284572598, 1.1253516563552464, 4.3471343888063512,
        -2.4025459467422121, -2.1708176372763099, 1.6665090945954870, -1.9197339981871877, 18.9149021356344598
    }, 10, 10);
  }
    
  private static Vector getB() {
    return new DenseVector(new double[] { 
        -0.552252, 0.038430, 0.058392, -1.234496, 1.240369, 0.373649, 0.505113, 0.503723, 1.215340, -0.391908
    });
  }
  
  private static Matrix getIllConditionedMatrix() {
    return reshape(new double[] {
        0.00695278043678842, 0.09911830022078683, 0.01309584636255063, 0.00652917453032394, 0.04337631487735064,
        0.14232165273321387, 0.05808722912361313, -0.06591965049732287, 0.06055771542862332, 0.00577423310349649,
        0.09911830022078683, 1.50071402418061428, 0.14988743575884242, 0.07195514527480981, 0.63747362341752722,
        1.30711819020414688, 0.82151609385115953, -0.72616125524587938, 1.03490136002022948, 0.12800239664439328,
        0.01309584636255063, 0.14988743575884242, 0.04068462583124965, 0.02147022047006482, 0.07388113580146650,
        0.58070223915076002, 0.11280336266257514, -0.21690068430020618, 0.04065087561300068, -0.00876895259593769,
        0.00652917453032394, 0.07195514527480981, 0.02147022047006482, 0.01140105250542524, 0.03624164348693958,
        0.31291554581393255, 0.05648457235205666, -0.11507583016077780, 0.01475756130709823, -0.00584453679519805,
        0.04337631487735064, 0.63747362341752722, 0.07388113580146649, 0.03624164348693959, 0.27491543200760571,
        0.73410543168748121, 0.36120630002843257, -0.36583546331208316, 0.41472509341940017, 0.04581458758255480,
        0.14232165273321387, 1.30711819020414666, 0.58070223915076002, 0.31291554581393255, 0.73410543168748121,
        9.02536073121807014, 1.25426385582883104, -3.16186335125594642, -0.19740140818905436, -0.26613760880058035,
        0.05808722912361314, 0.82151609385115953, 0.11280336266257514, 0.05648457235205667, 0.36120630002843257,
        1.25426385582883126, 0.48661058451606820, -0.57030511336562195, 0.49151280464818098, 0.04428280690189127,
       -0.06591965049732286, -0.72616125524587938, -0.21690068430020618, -0.11507583016077781, -0.36583546331208316,
       -3.16186335125594642, -0.57030511336562195, 1.16270815038078945, -0.14837898963724327, 0.05917203395002889,
        0.06055771542862331, 1.03490136002022926, 0.04065087561300068, 0.01475756130709823, 0.41472509341940023,
       -0.19740140818905436, 0.49151280464818103, -0.14837898963724327, 0.86693820682049716, 0.14089688752570340,
        0.00577423310349649, 0.12800239664439328, -0.00876895259593769, -0.00584453679519805, 0.04581458758255480,
       -0.26613760880058035, 0.04428280690189126, 0.05917203395002889, 0.14089688752570340, 0.02901858439788401
    }, 10, 10);    
  }
  
  private static Matrix getAsymmetricMatrix() {
    return reshape(new double[] {
        0.1586493402398226, -0.8668244036239467, 0.4335233711065471, -1.1025223577469705, 1.1344100191664601,
         -0.1399944083742454, 0.8879750333144295, -1.2139664527957903, 0.7154591081557057, -0.6320890356949669,
        -2.4546945723009581, 0.6354748667295935, -0.1931993736354496, -0.1210449542073575, -1.0668745874463414,
         0.6539061600017384, 2.4045520271091063,-0.3387572116155693, 0.1575188740437142, 1.1791073500243496,
        -0.6418745429181755, 0.6836410530720005, -1.2447493564334062, -1.8840081252627843, 0.5663864914859502,
         0.0819203791124956, 0.2004407540793239, 0.7350145066687849, 1.6525377683305262, -0.3156915229969668,
        -0.1866701463141060, -0.3929673444397022, -0.4440946700501859, 0.1366803303987421, -0.2138101381625466,
         0.5399874351478779, -1.0088091882703056, 0.0978023083150833, 1.8795777615527958, 0.3782417618354363,
        -0.4564752186043173, 0.4014814252832269, 1.9691150950571501, 0.2424686682362568, 1.0965758964799504,
         0.2751725463132324, -0.6652756564294597, -0.6256564536463288, 1.0332457212107204, -0.0330851504958215,
        -1.0402096493279287, -0.6850389655533707, -1.8896839974451625, 1.1533231017445102, -0.5387306882127710,
         0.0181850207098213, -0.2416652193929706, -0.9868171673047287, -1.5872573189377035, -0.8492253650362955,
         1.1949977792951225, 0.7901168665120927, 0.9832676055718492, -0.0752834029327588, 1.0555006468941126,
         0.6842531633106009, 0.2589700378872499, 0.3565253337268334, 0.1869608474650344, -0.1696524825242293,
         0.6919898638809949, -1.4937187919435133, 1.0039151841775080, -0.2580993333173019, 0.1243386429912411,
         1.3945380460721688,  0.3078165489952902, 1.1248734111054359,  0.5613308856003306, -0.9013329415656699,
        -0.9197179846787753,  0.1167372728291174, -0.7807620712716467,  0.2210918047063067, -0.4813869727362010,
         0.3870067788770671,  1.1974416632199159, 2.4676804711420330,  1.8492990765211168, -1.3089887830472471,
        -0.7587845769668021, -1.0354138253278353, -0.3907902473275445, -2.1292895670916168, -0.7544686049709807,
        -0.3431317172534703, 1.4959721683724390, 0.6004852467523584, 1.2140230344223786, 0.1279148299232956
    }, 20, 5);
  }
  
  private static Vector getSmallB() {
    return new DenseVector(new double[] {    
        0.114065955249272,
        0.953981568944476,
        -2.611106316607759,
        0.652190962446307,
        1.298055218126384,
    });
  }
  
  private static Matrix getLowrankSymmetricMatrix() {
    Matrix m = new DenseMatrix(5,5);
    Vector u = new DenseVector(new double[] {  
        -0.0364638798936962,
        1.0219291133418171,
        -0.5649933120375343,
        -1.0050553315595800,
        -0.5264178580727512
    });
    Vector v = new DenseVector(new double[] {
        -1.345847117891187,
        0.553386426498032,
        1.912020072696648,
        -0.820959934779948,
        1.223358044171859
    });

    return m.plus(u.cross(u)).plus(v.cross(v));
  }

  private static Matrix getLowrankAsymmetricMatrix() {
    Matrix m = new DenseMatrix(20,5);
    Vector u = new DenseVector(new double[] {  
        -0.0364638798936962,
        1.0219291133418171,
        -0.5649933120375343,
        -1.0050553315595800,
        -0.5264178580727512
    });
    Vector v = new DenseVector(new double[] {
        -1.345847117891187,
        0.553386426498032,
        1.912020072696648,
        -0.820959934779948,
        1.223358044171859
    });

    m.assignRow(0, u);
    m.assignRow(0, v);
    
    return m;
  }
  
  private static Matrix reshape(double[] values, int rows, int columns) {
    Matrix m = new DenseMatrix(rows, columns);
    int i = 0;
    for (double v : values) {
      m.set(i % rows, i / rows, v);
      i++;
    }
    return m;
  }
}
