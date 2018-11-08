var LineByLineReader = require('line-by-line'), lr = new LineByLineReader('shapemap1000000.txt');


var linenumber = 0;
var maxeps = 0;
lr.on('line', function (line) {
	//console.log(`Line ${linenumber}: ${line}`);
	if(linenumber % 3 === 2) {
		var dists = calcDists(line);
		eps = Math.abs( Math.max(...dists)-Math.min(...dists) );
		if(maxeps < eps)
			maxeps = eps;

		//console.log(`max: ${Math.max(...dists)}\nmin: ${Math.min(...dists)}\n`);
	}

	linenumber++;
	if(linenumber % 300000 === 0) {
		console.log("line " + linenumber);
		//lr.pause();
		//console.log(maxeps);
	}
});

lr.on('end', function () {
	console.log('done reading file.');
	console.log(maxeps);
});



/**
 * @param {string} data - string of 3d x, y, z coords separated by space
 * @return {array} dists - array of distances between each point 
 */
function calcDists(data) {
	data = data.split(' ');
	var coords = [];

	// split data into array of arrays of 3
	while(data.length > 1) // 1 because there is extra space at the end
		coords.push(data.splice(0, 3));

	var dists = [];
	for(let i = 0; i < coords.length-1; i++) {
		var xdiff = coords[i][0] - coords[i+1][0];
		var ydiff = coords[i][1] - coords[i+1][1];
		var zdiff = coords[i][2] - coords[i+1][2];

		dists.push( Math.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff) );
	}

	return dists;
}