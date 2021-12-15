import styled from '@emotion/styled';

export const SignupMainBox = styled.div`
	display: flex;
	flex-direction: row;
	justify-content: center;
	align-items: flex-start;
	width: 100%;
	@media (max-width: 82em) {
		flex-direction: column;
	}
`;

export const SignupLeftBox = styled.div`
	position: relative;
	display: flex;
	flex-direction: column;
	align-items: flex-end;
	justify-content: flex-start;
	width: 100%;
	height: 100%;
	min-height: 20%;
	padding: 0rem;
	margin: 2em 0 0 0;
	opacity: 1;
	text-align: center;
	@media (max-width: 82em) {
		align-items: center;
	}
`;

export const SignupRightBox = styled.div`
	position: relative;
	display: flex;
	width: 100%;
	font-family: arial;
	height: 100%;
	padding: 0rem;
	margin: 0rem;
	opacity: 1;

	@media (max-width: 82em) {
		justify-content: center;
	}
`;
