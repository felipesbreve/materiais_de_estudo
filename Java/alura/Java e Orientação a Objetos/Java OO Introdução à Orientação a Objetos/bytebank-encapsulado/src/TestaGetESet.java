
public class TestaGetESet {
	public static void main(String[] args) {
		Conta conta = new Conta(1337, 24226);
		
		System.out.println(conta.getNumero());
		
		Cliente felipe = new Cliente();
		
		//conta.titular = felipe;
		felipe.setNome("Felipe Breve");
		
		conta.setTitular(felipe);
		
		System.out.println(conta.getTitular().getNome());
		
		conta.getTitular().setProfissao("estudante");
		System.out.println(conta.getTitular().getProfissao());
	}

}
