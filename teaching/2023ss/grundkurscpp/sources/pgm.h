#ifndef PGM_H
#define PGM_H

#include <string>
#include <vector>

namespace IPK {
// reads a plain PGM image file from the given filename
// returns all read grey values in a twodimensional int array
// NOTE: very simple implementation that can't handle comment lines
std::vector<std::vector<int> > read_pgm(const std::string& filename);

// writes a plain PGM image file to the given filename
// grey values of image are read line by line from 2D int array
//
// The first index of data specifies the row, the second index specifies the
// column.
void write_pgm(const std::vector<std::vector<int> >& data,
               const std::string& filename);

}
#endif  // PGM_HH
