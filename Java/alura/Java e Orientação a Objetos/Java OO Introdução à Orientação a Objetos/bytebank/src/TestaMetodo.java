
public class TestaMetodo {
	
	public static void main(String[] args) {
		Conta minhaConta = new Conta();
		minhaConta.saldo = 100;
		minhaConta.deposita(50);
		System.out.println(minhaConta.saldo);
		
		boolean conseguiuRetirar = minhaConta.saca(20);
		System.out.println(minhaConta.saldo);
		System.out.println(conseguiuRetirar);
		
		Conta outraConta = new Conta();
		outraConta.deposita(1000);
		boolean sucessoTransferencia = outraConta.transfere(300, minhaConta);
		if(sucessoTransferencia) {
			System.out.println("Transferência realizada com sucesso!");
		} else {
			System.out.println("Falta dinheiro!");
		}
		System.out.println(outraConta.saldo);
		System.out.println(minhaConta.saldo);
		
		minhaConta.titular = "Felipe Breve";
		System.out.println(minhaConta.titular);
		
	}

}
