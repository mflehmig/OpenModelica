/*
 * This file is part of OpenModelica.
 *
 * Copyright (c) 1998-2019, Open Source Modelica Consortium (OSMC),
 * c/o Linköpings universitet, Department of Computer and Information Science,
 * SE-58183 Linköping, Sweden.
 *
 * All rights reserved.
 *
 * THIS PROGRAM IS PROVIDED UNDER THE TERMS OF THE BSD NEW LICENSE OR THE
 * GPL VERSION 3 LICENSE OR THE OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
 * ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
 * RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
 * ACCORDING TO RECIPIENTS CHOICE.
 *
 * The OpenModelica software and the OSMC (Open Source Modelica Consortium)
 * Public License (OSMC-PL) are obtained from OSMC, either from the above
 * address, from the URLs: http://www.openmodelica.org or
 * http://www.ida.liu.se/projects/OpenModelica, and in the OpenModelica
 * distribution. GNU version 3 is obtained from:
 * http://www.gnu.org/copyleft/gpl.html. The New BSD License is obtained from:
 * http://www.opensource.org/licenses/BSD-3-Clause.
 *
 * This program is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, EXCEPT AS
 * EXPRESSLY SET FORTH IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE
 * CONDITIONS OF OSMC-PL.
 *
 */

 /*! \file jacobian_symbolical.c
 */

#ifdef USE_PARJAC
  #include <omp.h>
#endif


#include "simulation/solver/jacobianSymbolical.h"

#ifdef USE_PARJAC
/** Allocate thread local Jacobians in case of OpenMP-parallel Jacobian computation.
 *
 * (symbolical only), used in IDA and Dassl.
 */
void allocateThreadLocalJacobians(DATA* data, ANALYTIC_JACOBIAN** jacColumns)
{
  int maxTh = omp_get_max_threads();
  *jacColumns = (ANALYTIC_JACOBIAN*) malloc(maxTh*sizeof(ANALYTIC_JACOBIAN));
  const int index = data->callback->INDEX_JAC_A;
  ANALYTIC_JACOBIAN* jac = &(data->simulationInfo->analyticJacobians[index]);

  //MS ToDo: Do we need this at any point?
  jac->sparsePattern = data->simulationInfo->analyticJacobians[data->callback->INDEX_JAC_A].sparsePattern;

  unsigned int columns = jac->sizeCols;
  unsigned int rows = jac->sizeRows;
  unsigned int sizeTmpVars = jac->sizeTmpVars;

  // ToDo: Benchmarks indicate that it is beneficial to initialize and malloc the jacColumns using a parallel for loop.
  // Rationale: The thread working on the data initializes the data and thus have it probably in its cache.
  unsigned int i;
  for (i = 0; i < maxTh; ++i) {
    (*jacColumns)[i].sizeCols = columns;
    (*jacColumns)[i].sizeRows = rows;
    (*jacColumns)[i].sizeTmpVars = sizeTmpVars;
    (*jacColumns)[i].tmpVars    = (double*) calloc(sizeTmpVars, sizeof(double));
    (*jacColumns)[i].resultVars = (double*) calloc(rows, sizeof(double));
    (*jacColumns)[i].seedVars   = (double*) calloc(columns, sizeof(double));
    (*jacColumns)[i].sparsePattern = data->simulationInfo->analyticJacobians[data->callback->INDEX_JAC_A].sparsePattern;
  }
}
#endif


/**
 * \brief Generic parallel computation of the colored Jacobian.
 *
 * Exploiting coloring and sparse structure. Used from DASSL and IDA solvers.
 * Only matrix storing format differs for them and therefore setJacElementFunc
 * is used to access matrix A.
 *
 * \param [in]      rows                Number of rows of jacobian.
 * \param [in]      columns             Number of columns of jacobian.
 * \param [in]      spp                 Pointer to sparse pattern.
 * \param [in/out]  matrixA             Internal data of solvers to store jacobian.
 * \param [in]      jacColumns          Number of colors (=number of columns for compressed structure) of jacobian.
 * \param [in]      data
 * \param [in]      threadData
 * \param [in]      setJacElementFunc   Function to set element (i,j) in matrix A.
 */
void genericColoredSymbolicJacobianEvaluation(int rows, int columns, SPARSE_PATTERN* spp,
                                              void* matrixA, ANALYTIC_JACOBIAN* jacColumns, DATA* data,
                                              threadData_t* threadData,
                                              void (*setJacElement)(int, int, int, double, void*, int))
{
#pragma omp parallel default(none) firstprivate(columns, rows) \
                                   shared(spp, matrixA, jacColumns, data, threadData, setJacElement)
{
#ifdef USE_PARJAC
  //  printf("My id = %d of max threads= %d\n", omp_get_thread_num(), omp_get_num_threads());
  ANALYTIC_JACOBIAN* t_jac = &(jacColumns[omp_get_thread_num()]);
#else
  ANALYTIC_JACOBIAN* t_jac = jacColumns;
#endif

  unsigned int i, j, currentIndex, nth;

#pragma omp for
  for (i=0; i < spp->maxColors; i++) {
    /* Set seed vector for current color */
    for (j=0; j < columns; j++) {
      if (spp->colorCols[j]-1 == i) {
        t_jac->seedVars[j] = 1;
      }
    }

    /* Evaluate with updated seed vector */
    data->callback->functionJacA_column(data, threadData, t_jac, NULL);
    /* Save jacobian elements in matrixA*/
    for (j=0; j < columns; j++) {
      if (t_jac->seedVars[j] == 1) {
        nth = spp->leadindex[j];
        while (nth < spp->leadindex[j+1]) {
          currentIndex = spp->index[nth];
          (*setJacElement)(currentIndex, j, nth, t_jac->resultVars[currentIndex], matrixA, rows);
          nth++;
        }
      }
    }

    /* Reset seed vector */
    for (j=0; j < columns; j++) {
      t_jac->seedVars[j] = 0;
    }
  }
} // omp parallel
}


/** Free ANALYTIC_JACOBIAN struct */
void freeAnalyticalJacobian(ANALYTIC_JACOBIAN* jacobian)
{
  free(jacobian->sparsePattern->leadindex);
  free(jacobian->sparsePattern->index);
  free(jacobian->sparsePattern->colorCols);
  free(jacobian->sparsePattern);
  free(jacobian->seedVars);
  free(jacobian->tmpVars);
  free(jacobian->resultVars);
  free(jacobian);
}
