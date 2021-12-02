#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "CSC.h"
#include "Triple.h"

// #include <tbb/scalable_allocator.h>

#include <random>
#include "utility.h"

namespace YusukeNagasaka{
using namespace std;

template <class IT, class NT>
class CSR
{ 
public:
    CSR():nnz(0), rows(0), cols(0),zerobased(true) {}
	CSR(IT mynnz, IT m, IT n):nnz(mynnz),rows(m),cols(n),zerobased(true)
	{
        // Constructing empty Csc objects (size = 0) are allowed (why wouldn't they?).
        assert(rows != 0);
        rowptr = my_malloc<IT>(rows + 1);
		if(nnz > 0) {
            colids = my_malloc<IT>(nnz);
            values = my_malloc<NT>(nnz);
        }
	}
    

    CSR (string filename);
    CSR (const CSC<IT,NT> & csc);   // CSC -> CSR conversion
    CSR (const CSR<IT,NT> & rhs);	// copy constructor
    CSR (const CSC<IT,NT> & csc, const bool transpose);
	CSR<IT,NT> & operator=(const CSR<IT,NT> & rhs);	// assignment operator
    bool operator==(const CSR<IT,NT> & rhs); // ridefinizione ==
    void shuffleIds(); // Randomly permutating column indices
    void sortIds(); // Permutating column indices in ascending order
    
    void make_empty()
    {
        if(nnz > 0) {
            my_free<IT>(colids);
            my_free<NT>(values);
            nnz = 0;
        }
        if(rows > 0) {
            my_free<IT>(rowptr);
            rows = 0;
        }
        cols = 0;	
    }
    
    ~CSR()
	{
        make_empty();
	}
    bool ConvertOneBased()
    {
        if(!zerobased)	// already one-based
            return false; 
        transform(rowptr, rowptr + rows + 1, rowptr, bind2nd(plus<IT>(), static_cast<IT>(1)));
        transform(colids, colids + nnz, colids, bind2nd(plus<IT>(), static_cast<IT>(1)));
        zerobased = false;
        return true;
    }
    bool ConvertZeroBased()
    {
        if (zerobased)
            return true;
        transform(rowptr, rowptr + rows + 1, rowptr, bind2nd(plus<IT>(), static_cast<IT>(-1)));
        transform(colids, colids + nnz, colids, bind2nd(plus<IT>(), static_cast<IT>(-1)));
        zerobased = true;
        return false;
    }
    bool isEmpty()
    {
        return ( nnz == 0 );
    }
    void Sorted();
    
	IT rows;	
	IT cols;
	IT nnz; // number of nonzeros
    
    IT *rowptr;
    IT *colids;
    NT *values;
    bool zerobased;
};

// copy constructor
template <class IT, class NT>
CSR<IT,NT>::CSR (const CSR<IT,NT> & rhs): nnz(rhs.nnz), rows(rhs.rows), cols(rhs.cols),zerobased(rhs.zerobased)
{
	if(nnz > 0)
	{
        values = my_malloc<NT>(nnz);
        colids = my_malloc<IT>(nnz);
        copy(rhs.values, rhs.values+nnz, values);
        copy(rhs.colids, rhs.colids+nnz, colids);
	}
	if ( rows > 0)
	{
        rowptr = my_malloc<IT>(rows + 1);
        copy(rhs.rowptr, rhs.rowptr+rows+1, rowptr);
	}
}

template <class IT, class NT>
CSR<IT,NT> & CSR<IT,NT>::operator= (const CSR<IT,NT> & rhs)
{
	if(this != &rhs)		
	{
		if(nnz > 0)	// if the existing object is not empty
		{
            my_free<IT>(colids);
            my_free<NT>(values);
		}
		if(rows > 0)
		{
            my_free<IT>(rowptr);
		}

		nnz	= rhs.nnz;
		rows = rhs.rows;
		cols = rhs.cols;
		zerobased = rhs.zerobased;
		if(rhs.nnz > 0)	// if the copied object is not empty
		{
            values = my_malloc<NT>(nnz);
            colids = my_malloc<IT>(nnz);
            copy(rhs.values, rhs.values+nnz, values);
            copy(rhs.colids, rhs.colids+nnz, colids);
		}
		if(rhs.cols > 0)
		{
            rowptr = my_malloc<IT>(rows + 1);
            copy(rhs.rowptr, rhs.rowptr+rows+1, rowptr);
		}
	}
	return *this;
}

//! Construct a CSR object from a CSC
//! Accepts only zero based CSC inputs
template <class IT, class NT>
CSR<IT,NT>::CSR(const CSC<IT,NT> & csc):nnz(csc.nnz), rows(csc.rows), cols(csc.cols),zerobased(true)
{
    rowptr = my_malloc<IT>(rows + 1);
    colids = my_malloc<IT>(nnz);
    values = my_malloc<NT>(nnz);

    IT *work = my_malloc<IT>(rows);
    std::fill(work, work+rows, (IT) 0); // initilized to zero
   
    	for (IT k = 0 ; k < nnz ; ++k)
    	{
        	IT tmp =  csc.rowids[k];
        	work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
	}

	if(nnz > 0)
	{
		rowptr[rows] = CumulativeSum (work, rows);		// cumulative sum of w
       	 	copy(work, work+rows, rowptr);

		IT last;
        	for (IT i = 0; i < cols; ++i) 
        	{
       	     		for (IT j = csc.colptr[i]; j < csc.colptr[i+1] ; ++j)
            		{
				colids[ last = work[ csc.rowids[j] ]++ ]  = i ;
				values[last] = csc.values[j] ;
            		}
        	}
	}
    my_free<IT>(work);
}

template <class IT, class NT>
CSR<IT,NT>::CSR(const CSC<IT,NT> & csc, const bool transpose):nnz(csc.nnz), rows(csc.rows), cols(csc.cols),zerobased(true)
{
    if (!transpose) {
        rowptr = my_malloc<IT>(rows + 1);
        colids = my_malloc<IT>(nnz);
        values = my_malloc<NT>(nnz);

        IT *work = my_malloc<IT>(rows);
        std::fill(work, work+rows, (IT) 0); // initilized to zero
   
    	for (IT k = 0 ; k < nnz ; ++k)
            {
                IT tmp =  csc.rowids[k];
                work [ tmp ]++ ;		// row counts (i.e, w holds the "row difference array")
            }

        if(nnz > 0) 
            {
                rowptr[rows] = CumulativeSum (work, rows);		// cumulative sum of w
                copy(work, work+rows, rowptr);

                IT last;
                for (IT i = 0; i < cols; ++i) 
                    {
                        for (IT j = csc.colptr[i]; j < csc.colptr[i+1] ; ++j)
                            {
                                colids[ last = work[ csc.rowids[j] ]++ ]  = i ;
                                values[last] = csc.values[j] ;
                            }
                    }
            }
        my_free<IT>(work);
    }
    else {
        rows = csc.cols;
        cols = csc.rows;
        rowptr = my_malloc<IT>(rows + 1);
        colids = my_malloc<IT>(nnz);
        values = my_malloc<NT>(nnz);

        for (IT k = 0; k < rows + 1; ++k) {
            rowptr[k] = csc.colptr[k];
        }
        for (IT k = 0; k < nnz; ++k) {
            values[k] = csc.values[k];
            colids[k] = csc.rowids[k];
        }
    }
}

// check if sorted within rows?
template <class IT, class NT>
void CSR<IT,NT>::Sorted()
{
	bool sorted = true;
	for(IT i=0; i< rows; ++i)
	{
		sorted &= my_is_sorted (colids + rowptr[i], colids + rowptr[i+1], std::less<IT>());
    }
}

template <class IT, class NT>
bool CSR<IT,NT>::operator==(const CSR<IT,NT> & rhs)
{
    bool same;
    if(nnz != rhs.nnz || rows  != rhs.rows || cols != rhs.cols) {
        printf("%d:%d, %d:%d, %d:%d\n", nnz, rhs.nnz, rows, rhs.rows, cols, rhs.cols);
        return false;
    }  
    if (zerobased != rhs.zerobased) {
        IT *tmp_rowptr = my_malloc<IT>(rows + 1);
        IT *tmp_colids = my_malloc<IT>(nnz);
        if (!zerobased) {
            for (int i = 0; i < rows + 1; ++i) {
                tmp_rowptr[i] = rowptr[i] - 1;
            }
            for (int i = 0; i < nnz; ++i) {
                tmp_colids[i] = colids[i] - 1;
            }
            same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rhs.rowptr); 
            same = same && std::equal(tmp_colids, tmp_colids + nnz, rhs.colids);
        }
        else if (!rhs.zerobased) {
            for (int i = 0; i < rows + 1; ++i) {
                tmp_rowptr[i] = rhs.rowptr[i] - 1;
            }
            for (int i = 0; i < nnz; ++i) {
                tmp_colids[i] = rhs.colids[i] - 1;
            }
            same = std::equal(tmp_rowptr, tmp_rowptr + rows + 1, rowptr); 
            same = same && std::equal(tmp_colids, tmp_colids + nnz, colids);

        }
        my_free<IT>(tmp_rowptr);
        my_free<IT>(tmp_colids);
    }
    else {
        same = std::equal(rowptr, rowptr+rows+1, rhs.rowptr); 
        same = same && std::equal(colids, colids+nnz, rhs.colids);
    }
    
    bool samebefore = same;
    ErrorTolerantEqual<NT> epsilonequal(EPSILON);
    same = same && std::equal(values, values+nnz, rhs.values, epsilonequal );
    if(samebefore && (!same)) {
#ifdef DEBUG
        vector<NT> error(nnz);
        transform(values, values+nnz, rhs.values, error.begin(), absdiff<NT>());
        vector< pair<NT, NT> > error_original_pair(nnz);
        for(IT i=0; i < nnz; ++i)
            error_original_pair[i] = make_pair(error[i], values[i]);
        if(error_original_pair.size() > 10) { // otherwise would crush for small data
            partial_sort(error_original_pair.begin(), error_original_pair.begin()+10, error_original_pair.end(), greater< pair<NT,NT> >());
            cout << "Highest 10 different entries are: " << endl;
            for(IT i=0; i < 10; ++i)
                cout << "Diff: " << error_original_pair[i].first << " on " << error_original_pair[i].second << endl;
        }
        else {
            sort(error_original_pair.begin(), error_original_pair.end(), greater< pair<NT,NT> >());
            cout << "Highest different entries are: " << endl;
            for(typename vector< pair<NT, NT> >::iterator it=error_original_pair.begin(); it != error_original_pair.end(); ++it)
                cout << "Diff: " << it->first << " on " << it->second << endl;
        }
#endif
            }
    return same;
}


template <class IT, class NT>
void CSR<IT,NT>::shuffleIds()
{
    mt19937_64 mt(0);
    for (IT i = 0; i < rows; ++i) {
        IT offset = rowptr[i];
        IT width = rowptr[i + 1] - rowptr[i];
        uniform_int_distribution<IT> rand_scale(0, width - 1);
        for (IT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IT target = rand_scale(mt);
            IT tmpId = colids[offset + target];
            NT tmpVal = values[offset + target];
            colids[offset + target] = colids[j];
            values[offset + target] = values[j];
            colids[j] = tmpId;
            values[j] = tmpVal;
        }
    }
}


}