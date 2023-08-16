#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <algorithm>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

// Write down the kernels here


__global__ void fill_worklist(int numReq,int *fac_worklist,int *fac_req_start_id,int *req_id,int *req_cen,int *req_fac,int *centre_firstfac_global_id) //Each Thread of the kernel processes a request and fills it in the worklist of corresponding facility
{
	int reqid,fac_id,idx;
	reqid = (blockIdx.x*blockDim.x) + threadIdx.x; //Calculates the request id corresponding to this thread

	if(reqid<numReq) //if request id is less than total no. of requests
	{
		fac_id = centre_firstfac_global_id[req_cen[reqid]]+req_fac[reqid]; //Calculates the global id of the facility corresponding to this request
		idx = atomicInc((unsigned int *) &fac_req_start_id[fac_id],100002); //Gets the index of where in fac_worklist this request should i.e location corresponding to fac_id and atomically increments the marker where the next request processed for this fac_id should go
		fac_worklist[idx]=reqid; //Puts the request reqid at the index idx
	}
}

__global__ void evaluate_worklist(int numFac,int *fac_worklist,int *fac_req_start_id,int *req_id,int *req_cen,int *req_fac,int *centre_firstfac_global_id,int *capacity,int *tot_reqs_forfac,int *req_start,int *req_slots,int *succ_reqs,int *slots) //Each thread of the kernel processes a facility whose requests(already sorted earlier) are evaluated for their success/failure
{
	int fac_id,slotid,reqid,startslot,flag;
	fac_id = (blockIdx.x*blockDim.x) + threadIdx.x; //Calculates the global id of facility corresponding to this thread
	if(fac_id<numFac) //if fac_id is less than total no. of facilities
	{
		slotid = 24*fac_id; //Index of slot 0 corresponding to fac_id in slots array
		
		for(int i=0;i<tot_reqs_forfac[fac_id];i++) //Iterate through all requests corresponding to fac_id one by one (Requests are already sorted in the worklist)
		{
			reqid = fac_worklist[fac_req_start_id[fac_id]+i]; //Get the Request if of the request to be analyzed
			startslot = slotid+req_start[reqid]-1; //Gets the start slot index corresponding to request in slots (-1) is for indexing as Slots vary from 1 to 24 and thier indexes will vary from 0 to 23
			flag=1; //Flag is true initially
			for(int j=0;j<req_slots[reqid];j++) //Iterates through all requested slots to verify if they are within capapcity
			{
				if(slots[startslot+j]==capacity[fac_id]) //If any slot is full sets flag to false/0 and breaks
				{
					flag=0;
					break;
				}
			}
			
			if(flag) //If flag reamains true/1 means slots are available hence considers this request successful and allots this request its requsted slot by increasing current levels of requested slots by 1
			{
				for(int j=0;j<req_slots[reqid];j++)
				{
					slots[startslot+j]++;
				}
				atomicInc((unsigned int *) &succ_reqs[req_cen[reqid]],100002); //Since this request is succesful,Atomically updates/increments the number of successful requests corresponding to this facility's centre
			}

		}
	}
}

//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 

    int *centre_firstfac_global_id; //Added variable stores global id for a centre's first facility

    centre_firstfac_global_id = (int*)malloc(N * sizeof (int));

    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      centre_firstfac_global_id[i]=k1; //Global id for a centre i's first facility will be k1

      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    int *tot_reqs_forfac; //Stores total no. of requests for each facility at their global id index
    tot_reqs_forfac = (int *) malloc((k1) * sizeof(int));
    memset(tot_reqs_forfac, 0, k1*sizeof(int));

    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;

       tot_reqs_forfac[centre_firstfac_global_id[req_cen[j]]+req_fac[j]]+=1;  //Increments no. of requests for corresponding facility(using its globl id as index)
    }
		


    //*********************************
    // Call the kernels here
    //********************************

    int *d_req_id, *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots; //Corresponding arrays on Device/GPU

    cudaMalloc(&d_req_id, R*sizeof(int));
    cudaMalloc(&d_req_cen, R*sizeof(int));
    cudaMalloc(&d_req_fac, R*sizeof(int));
    cudaMalloc(&d_req_start, R*sizeof(int));
    cudaMalloc(&d_req_slots, R*sizeof(int));

    cudaMemcpy(d_req_id, req_id, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac, req_fac, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start, req_start, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots, req_slots, R*sizeof(int), cudaMemcpyHostToDevice);

    int *fac_req_start_id, *fac_worklist, *d_fac_worklist;

    fac_worklist = (int *) malloc((R) * sizeof(int)); //Contains all the requests split into sections/bins corresponding to their facilities such that all requests corresponding to a facility are contiguous
    cudaMalloc(&d_fac_worklist, R*sizeof(int));

    fac_req_start_id = (int *) malloc((k1) * sizeof(int)); //Stores start index corresponding to a facility in the worklist 

    int curr=0;
    for(int i=0;i<k1;i++) //Computes start index corresponding to a facility in the worklist
    {
	fac_req_start_id[i]=curr;
	curr+=tot_reqs_forfac[i];
    }
    
    int *d_tot_reqs_forfac; //Updates corresponding host arrays onto device arrays
    cudaMalloc(&d_tot_reqs_forfac, k1*sizeof(int));
    cudaMemcpy(d_tot_reqs_forfac, tot_reqs_forfac, k1*sizeof(int), cudaMemcpyHostToDevice);

    int *d_fac_req_start_id; //Updates corresponding host arrays onto device arrays
    cudaMalloc(&d_fac_req_start_id, k1*sizeof(int));
    cudaMemcpy(d_fac_req_start_id, fac_req_start_id, k1*sizeof(int), cudaMemcpyHostToDevice);

    int *d_centre_firstfac_global_id; //Updates corresponding host arrays onto device arrays
    cudaMalloc(&d_centre_firstfac_global_id, N*sizeof(int));
    cudaMemcpy(d_centre_firstfac_global_id, centre_firstfac_global_id, N*sizeof(int), cudaMemcpyHostToDevice);

    int nBlocks=ceil((float)(R)/1024); //No. of blocks for the fill_worklist kernel such that each thread processes a request

    fill_worklist<<<nBlocks,1024>>>(R,d_fac_worklist,d_fac_req_start_id,d_req_id,d_req_cen,d_req_fac,d_centre_firstfac_global_id); //Calls the fill_worklist kernels whose each threads process a request and fill it in the worklist region of their corresponding facility
    cudaDeviceSynchronize();

    cudaMemcpy(fac_worklist,d_fac_worklist, R*sizeof(int), cudaMemcpyDeviceToHost); //Copies back the updated worklist from device to host

    for(int i=0;i<k1;i++)
    {
	sort(&fac_worklist[fac_req_start_id[i]],&fac_worklist[fac_req_start_id[i]+tot_reqs_forfac[i]]); //Sorts the worklist area contining requests corresponding to facility with global id i
    }

    int *d_succ_reqs,*d_capacity,*d_facslots;

    cudaMalloc(&d_succ_reqs, N*sizeof(int)); //Stores successful request corresponding to each centre
    cudaMalloc(&d_capacity, k1*sizeof(int)); //Contains capacity of each facility
    cudaMalloc(&d_facslots, 24*k1*sizeof(int)); //Stores the status of 24 slots corresponding to each facility

    cudaMemset(d_succ_reqs, 0, N*sizeof(int));
    cudaMemset(d_facslots, 0, 24*k1*sizeof(int));

    cudaMemcpy(d_capacity, capacity, k1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fac_req_start_id, fac_req_start_id, k1*sizeof(int), cudaMemcpyHostToDevice); //Using host array restores the device fac_req_start_id which was changed by fill_worklist kernel while filling the worklist
    cudaMemcpy(d_fac_worklist, fac_worklist, R*sizeof(int), cudaMemcpyHostToDevice);

    nBlocks=ceil((float)(k1)/1024); //No. of blocks for the evaluate_worklist kernel such that each thread processes a facility

    evaluate_worklist<<<nBlocks,1024>>>(k1,d_fac_worklist,d_fac_req_start_id,d_req_id,d_req_cen,d_req_fac,d_centre_firstfac_global_id,d_capacity,d_tot_reqs_forfac,d_req_start,d_req_slots,d_succ_reqs,d_facslots); //Calls the fill_worklist kernels whose each threads process a facility whose requests(already sorted earlier) are evaluated for their success/failure
    cudaDeviceSynchronize();

    cudaMemcpy(succ_reqs,d_succ_reqs, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i=0;i<N;i++) //Calculates Total no. of success by iterating over each centre's succesful requests
    {
	success+=succ_reqs[i];
    }

    fail = R-success; //Total Failed requests is Total requests - Total successful requests

    cudaFree(d_req_id);
    cudaFree(d_req_cen);
    cudaFree(d_req_fac);
    cudaFree(d_req_start);
    cudaFree(d_req_slots);
    cudaFree(d_tot_reqs_forfac);
    cudaFree(d_fac_req_start_id);
    cudaFree(d_centre_firstfac_global_id);
    cudaFree(d_succ_reqs);
    cudaFree(d_capacity);
    cudaFree(d_facslots);

    free(tot_reqs_forfac);
    free(fac_req_start_id);
    free(centre_firstfac_global_id);

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}