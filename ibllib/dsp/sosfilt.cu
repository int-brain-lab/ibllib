const int  n_signals = 384, n_samples = 65700, n_sections = 2;


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void sosfilt(const float *sos, float *x, const float *zi){
    int s, t;
    float x_cur, x_new;
    float zi_t[n_sections*2], sos_t[n_sections*6];
    int tid = threadIdx.x;
    int nt = n_samples;
    int ns = n_sections;

    // Index of signal to be filtered
    int signal_id = blockIdx.x * blockDim.x + tid;

    if (signal_id >= n_signals) return;

    // Move arrays to local thread memory
    for(s=0;s<ns*2;s++){
        zi_t[s] = zi[signal_id*n_sections*2 + s];
    }

    for(s=0;s<ns*6;s++){
        sos_t[s] = sos[s];
    }

    // Filtering
    for(t=0;t<nt;t++){
        x_cur = x[signal_id*nt+t];
        for(s=0;s<ns;s++){
            x_new = sos_t[s*6] * x_cur + zi_t[s*2];
            zi_t[s*2] = sos_t[s*6+1] * x_cur - sos_t[s*6+4] * x_new + zi_t[s*2+1];
            zi_t[s*2+1] = sos_t[s*6+2] * x_cur - sos_t[s*6+5] * x_new;
            x_cur = x_new;
        }
        x[signal_id*nt+t] = x_cur;
    }
}
