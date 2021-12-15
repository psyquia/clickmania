import React from 'react';
import { Link } from 'react-router-dom';
import { Navbar } from 'components';
import { css } from '@emotion/css';
import StoreMain from './StoreMain';

const Store = () => (
	<>
		<Navbar />
		<div
			className={css`
				margin-top: 30px;
				width: 100%;
				display: flex;
				flex-direction: column;
				justify-content: center;
				align-items: center;
				text-align: center;
			`}
		>
			<StoreMain />
			<Link to="/">Back to homepage</Link>
		</div>
	</>
);

export default Store;
