// CommonJS PCN Reference Implementation
// From user input: reference skeleton for algorithm validation
// This is NOT the actual implementation, but a guide for understanding the math

function tanh(x) { return Math.tanh(x); }
function dtanh_from_x(x) { const t = Math.tanh(x); return 1 - t * t; }
function zeros(n) { return Array(n).fill(0); }

// W: outDim x inDim
function matvec(W, v) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
        let s = 0;
        const row = W[i];
        for (let j = 0; j < row.length; j++) s += row[j] * v[j];
        out[i] = s;
    }
    return out;
}

function matTvec(W, v) {
    // W^T * v where W: outDim x inDim, v: outDim
    const inDim = W[0].length;
    const out = new Array(inDim).fill(0);
    for (let i = 0; i < W.length; i++) {
        const row = W[i];
        const vi = v[i];
        for (let j = 0; j < inDim; j++) out[j] += row[j] * vi;
    }
    return out;
}

function add(a, b) { return a.map((x, i) => x + b[i]); }
function sub(a, b) { return a.map((x, i) => x - b[i]); }
function hadamard(a, b) { return a.map((x, i) => x * b[i]); }
function scale(a, s) { return a.map(x => x * s); }

function outerAdd(W, a, b, lr) {
    // W += lr * (a outer b), where a: outDim, b: inDim
    for (let i = 0; i < W.length; i++) {
        const row = W[i];
        const ai = a[i] * lr;
        for (let j = 0; j < row.length; j++) row[j] += ai * b[j];
    }
}

class PCN {
    // dims: [d0, d1, ... dL]
    constructor(dims) {
        this.dims = dims;
        this.L = dims.length - 1;
        // W[l] predicts layer (l-1) from l: shape d_{l-1} x d_l
        this.W = [];
        this.b = [];
        // b[l-1] shape d_{l-1}
        for (let l = 1; l <= this.L; l++) {
            const outDim = dims[l - 1];
            const inDim = dims[l];
            const Wl = Array.from(
                { length: outDim },
                () => Array.from({ length: inDim }, () => (Math.random() - 0.5) * 0.1)
            );
            this.W[l] = Wl;
            this.b[l - 1] = zeros(outDim);
        }
    }

    // one sample training step
    trainSample(x0, xL_target, { relaxSteps = 20, alpha = 0.05, eta = 0.01, clampTop = true } = {}) {
        // states x[l]
        const x = [];
        for (let l = 0; l <= this.L; l++) x[l] = zeros(this.dims[l]);
        // clamp input
        x[0] = x0.slice();
        // optional clamp output label
        if (clampTop) x[this.L] = xL_target.slice();

        // errors eps[l] for layer l
        const eps = [];
        for (let l = 0; l <= this.L; l++) eps[l] = zeros(this.dims[l]);

        // relaxation
        for (let t = 0; t < relaxSteps; t++) {
            // compute eps for layers 0..L-1
            for (let l = 1; l <= this.L; l++) {
                const fx = x[l].map(tanh);
                const mu = add(matvec(this.W[l], fx), this.b[l - 1]);
                eps[l - 1] = sub(x[l - 1], mu);
            }

            // update internal states 1..L-1 (do not update clamped layers)
            for (let l = 1; l <= this.L - 1; l++) {
                const d1 = scale(eps[l], -1);
                const back = matTvec(this.W[l + 1], eps[l - 1]); // (W^{l+1})^T eps^{l-1}
                const df = x[l].map(dtanh_from_x);
                const d2 = hadamard(back, df);
                const dx = add(d1, d2);
                x[l] = add(x[l], scale(dx, alpha));
            }
        }

        // final eps for weight updates
        for (let l = 1; l <= this.L; l++) {
            const fx = x[l].map(tanh);
            const mu = add(matvec(this.W[l], fx), this.b[l - 1]);
            eps[l - 1] = sub(x[l - 1], mu);
            // Hebbian-ish: W[l] += eta * (eps[l-1] outer f(x[l]))
            outerAdd(this.W[l], eps[l - 1], fx, eta);
            // bias += eta * eps
            this.b[l - 1] = add(this.b[l - 1], scale(eps[l - 1], eta));
        }

        return { x, eps };
    }
}

module.exports = { PCN };
