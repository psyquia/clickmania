import React from 'react';
import { Navbar } from 'components';
import { css } from '@emotion/css';

const Explore = () => (
	<>
		<Navbar />
		<div
			className={css`
				width: 100%;
				display: flex;
				flex-direction: column;
				justify-content: center;
				align-items: center;
				text-align: center;
			`}
		>
			<h1>This is Explore page</h1>
		</div>
	</>
);

export default Explore;
