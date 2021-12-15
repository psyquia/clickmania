import React, { useState, useEffect } from 'react';
import { css } from '@emotion/css';
import { getAllCollections } from 'api/apiCalls';
import Collection from './Collection.js';

const StoreMain = () => {
	const [collections, setCollections] = useState([]);

	useEffect(() => {
		handleGetCollections();
	}, []);

	const handleGetCollections = async () => {
		const col = await getAllCollections();
		if (!col.length) return;
		// reverse order
		col.sort((a, b) => b.collectionID - a.collectionID);
		setCollections(col);
	};

	return (
		<div
			className={css`
				width: 80%;
				max-width: 80em;
			`}
		>
			<h1>Loot Boxes</h1>
			{collections &&
				collections.map((collection) => <Collection key={collection.collectionID} collection={collection} />)}
		</div>
	);
};

export default StoreMain;
