__kernel void Add1(__global float* delta, __global float* start_point, __global float* c, int size, __global float* tmp)
{
    // Find position in global arrays.
	//delta przedzial na 1 karte
	//delta/size wielkosc przedzialu kazdego watku
	int temp = *delta;
	*delta = (*delta)/size;
    int id = get_global_id(0);
	
	
	if(id == 0) *c = 0;

	float dx = *delta;//0.0001;
	
	tmp[id] = 0;
	
		for(float i = *start_point + id* (*delta); i < *start_point + ((id+1) * (*delta) -dx ); i+=dx){
				tmp[id] += ((i + (i+dx))*dx/2);
				//tmp[id] += ((i*i + ((i+dx)*(i+dx)))*dx/2);
				//tmp[id] += 1;
		}

		 barrier ( CLK_GLOBAL_MEM_FENCE );
	if (id == 0)
	{
		for(int i = 0; i<size; i++){
			*c += tmp[i];
		}
	}
}