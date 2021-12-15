import React from 'react';
import './Filterbar.css';

const Filterbar = ({ handleFilterChange, filterQuery }) => (
	<div className="filter-container">
		<input
			type="filter"
			className="filter-input"
			onChange={handleFilterChange}
			value={filterQuery}
			placeholder="Filter..."
		/>
	</div>
);

export default Filterbar;
