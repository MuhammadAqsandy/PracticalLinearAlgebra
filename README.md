# 📚 Practical Linear Algebra for Data Science (O'Reilly)

> **Tugas 3 — Machine Learning & Deep Learning | Universitas Telkom Bandung**  
> Code Reproduction + Theoretical Deep-Dive from *Practical Linear Algebra* (O'Reilly)

---

## 👤 Author
| NIM | Nama |
|-----|------|
| 1103220214 | Aqsandy |

---

## 📖 About This Repository

This repository contains reproduced code and theoretical explanations for every chapter of **Practical Linear Algebra for Data Science** (O'Reilly). Each chapter is presented as a Jupyter Notebook with:
- ✅ Reproduced and working Python code
- 📝 Theoretical explanation of key concepts
- 🔗 Connections to Machine Learning applications

---

## 📂 Repository Structure

```
PracticalLinearAlgebra/
├── 01.Chapter01_Introduction_to_Linear_Algebra.ipynb
├── 02.Chapter02_Vectors_Part_1_Operations_and_Geometry.ipynb
├── 03.Chapter03_Vectors_Part_2_Spaces_and_Projections.ipynb
├── 04.Chapter04_Vector_Applications.ipynb
├── 05.Chapter05_Matrices_Part_1_Basics_and_Operations.ipynb
├── 06.Chapter06_Matrices_Part_2_Advanced_Properties.ipynb
├── 07.Chapter07_Matrix_Applications.ipynb
├── 08.Chapter08_Matrix_Inverse_and_Linear_Systems.ipynb
├── 09.Chapter09_Orthogonal_Matrices_and_QR_Decomposition.ipynb
├── 10.Chapter10_Row_Reduction_and_LU_Decomposition.ipynb
├── 11.Chapter11_General_Least_Squares.ipynb
├── 12.Chapter12_Least_Squares_Applications.ipynb
├── 13.Chapter13_Eigendecomposition.ipynb
├── 14.Chapter14_Singular_Value_Decomposition_SVD.ipynb
├── 15.Chapter15_Eigendecomposition_and_SVD_Applications.ipynb
└── README.md
```

---

## 📑 Chapter Summaries

### Chapter 1 — Introduction to Linear Algebra
Introduces the four fundamental objects: **scalars** (single numbers), **vectors** (1D arrays), **matrices** (2D grids), and **tensors** (multi-dimensional arrays). NumPy is established as the primary computation tool. Understanding these structures is essential since virtually all ML algorithms operate on matrices and vectors.

---

### Chapter 2 — Vectors Part 1: Operations and Geometry
Covers core vector operations: **addition, subtraction, scalar multiplication, dot product**, and **vector norms** (L1, L2). The dot product is the engine behind neural network computations, and vector norms quantify magnitude. Unit vectors preserve direction with length = 1, widely used in normalization.

---

### Chapter 3 — Vectors Part 2: Spaces and Projections
Explores **linear combinations, linear independence, span**, and **orthogonality**. Orthogonal vectors (dot product = 0) are perpendicular and fundamental to PCA and QR decomposition. **Vector projection** decomposes one vector along another, underpinning least squares regression.

---

### Chapter 4 — Vector Applications
Applies vector concepts to data science problems: **Pearson correlation** (cosine similarity of mean-centered vectors), **document similarity** in NLP, **mean centering** for preprocessing, **weighted averages** (core of neural network layers), and **moving averages** for time series smoothing.

---

### Chapter 5 — Matrices Part 1: Basics and Operations
Introduces matrix creation, element-wise **addition/subtraction**, and **matrix multiplication** (the central ML operation). Key insight: AB ≠ BA (not commutative). The **transpose** flips rows and columns, critical for covariance computation. Special matrices (identity, diagonal, triangular) simplify many algorithms.

---

### Chapter 6 — Matrices Part 2: Advanced Properties
Deep-dives into **determinant** (measures volume scaling; zero = singular/non-invertible), **trace** (sum of diagonal = sum of eigenvalues), **matrix rank** (dimensionality of column space), and **Frobenius norm** (measures overall matrix size, used in weight decay regularization).

---

### Chapter 7 — Matrix Applications
Demonstrates matrices as **linear transformations** (rotation, shear, scaling) and as **datasets** (rows = samples, columns = features). The **covariance matrix** captures feature relationships and is the input to PCA. Linear systems Ax = b represent regression and optimization problems.

---

### Chapter 8 — Matrix Inverse and Linear Systems
Covers **matrix inversion** and solving linear systems. A matrix is invertible only when det ≠ 0 (full rank). `np.linalg.solve()` is preferred over explicit inverse for **numerical stability**. The **Moore-Penrose pseudoinverse** extends inversion to non-square/singular matrices. The **condition number** measures sensitivity to perturbations.

---

### Chapter 9 — Orthogonal Matrices and QR Decomposition
Orthogonal matrices (Q^T Q = I) preserve lengths and angles — rotations and reflections. **Gram-Schmidt** builds orthonormal bases from arbitrary vectors. **QR decomposition** (A = QR) is numerically stable and used for solving linear systems, computing eigenvalues, and least-squares fitting.

---

### Chapter 10 — Row Reduction and LU Decomposition
**Gaussian elimination** reduces systems to row echelon form. **LU decomposition** (PA = LU) factors into lower and upper triangular matrices. The main efficiency advantage: once LU is computed, solving for multiple right-hand sides costs O(n²) instead of O(n³) — crucial for iterative methods.

---

### Chapter 11 — General Least Squares
When more equations than unknowns exist (overdetermined), no exact solution exists. **Least squares** minimizes the sum of squared residuals. The **normal equations** (A^T A x = A^T b) derive the optimal solution. `np.linalg.lstsq()` is preferred for stability. This is the **mathematical foundation of linear regression**.

---

### Chapter 12 — Least Squares Applications
Extends least squares to **polynomial regression** (polynomial feature design matrix), **multiple regression** (many features), and **Ridge regression** (L2 regularization: λ||w||² penalty). Shows overfitting with high-degree polynomials, motivating regularization to improve generalization.

---

### Chapter 13 — Eigendecomposition
Eigenvectors are special vectors that only scale under matrix multiplication: **Av = λv**. **Diagonalization** A = VDV⁻¹ makes matrix powers efficient. Symmetric matrices (spectral theorem) always yield real eigenvalues and orthogonal eigenvectors. Key identity: trace = Σλᵢ, determinant = Πλᵢ.

---

### Chapter 14 — Singular Value Decomposition (SVD)
**SVD** (A = UΣV^T) is the most general matrix factorization, working for any shape. Singular values measure "importance" of each component. **Truncated SVD** (keeping k largest singular values) gives the best rank-k approximation (Eckart-Young theorem) — the basis of image compression, PCA, and recommender systems.

---

### Chapter 15 — Eigendecomposition and SVD Applications
Brings everything together: **PCA from scratch** (eigenvectors of covariance matrix for dimensionality reduction), **Eigenfaces** (face recognition via SVD of image data), **SVD-based collaborative filtering** (recommender systems), and **spectral clustering** (graph Laplacian eigenvectors for non-linear cluster discovery).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Primary language |
| NumPy | Matrix & vector operations |
| SciPy | Advanced decompositions (LU, etc.) |
| Matplotlib | Visualizations |
| Scikit-learn | ML datasets & comparisons |
| SymPy | Symbolic math (row reduction) |
| Jupyter Notebook | Interactive notebooks |

---

## 🚀 How to Run

```bash
# Clone this repo
git clone https://github.com/<your-username>/PracticalLinearAlgebra.git
cd PracticalLinearAlgebra

# Install dependencies
pip install numpy scipy matplotlib scikit-learn sympy scikit-image jupyter

# Launch Jupyter
jupyter notebook
```

---

## 📚 Reference

**Book:** *Practical Linear Algebra for Data Science* — Mike X Cohen (O'Reilly, 2022)  
**Official repo:** [https://github.com/mikexcohen/LinAlg4DataScience](https://github.com/mikexcohen/LinAlg4DataScience)

---

> *All work is original and adheres to academic integrity standards.*
